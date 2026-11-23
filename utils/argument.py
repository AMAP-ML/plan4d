"""
Utility functions for parsing command line arguments.
"""

import argparse
import os
import pathlib
import textwrap

from models.stitching_layer_builder import parse_conv_spec

DEFAULT_DA3_STITCHING_SPEC = "conv3d_k5x3x3_o1536_s1x2x2_p2x1x1"

# -------------------------
# Types / helpers
# -------------------------


def parse_dataset(arg: str) -> tuple[str, pathlib.Path]:
    """
    Convert a CLI token of the form NAME:ROOT into (name, root)
    and validate a few basic things.
    """
    try:
        name, root = arg.split(":", 1)
    except ValueError:  # no “:”
        raise argparse.ArgumentTypeError(
            "Dataset must be NAME:PATH, e.g. dl3dv:/data/dl3dv"
        )

    root_path = pathlib.Path(root).expanduser()
    if not root_path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {root_path}")

    return name, root_path


def parse_manifest_spec(arg: str) -> tuple[str, pathlib.Path]:
    try:
        name, manifest_path = arg.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Manifest must be NAME:PATH, e.g. point_odyssey:/path/to/point_odyssey_train.jsonl"
        ) from exc

    path = pathlib.Path(manifest_path).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")

    return name, path


def parse_existing_file(arg: str) -> str:
    path = pathlib.Path(arg).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    return str(path)


def parse_existing_path(arg: str) -> str:
    path = pathlib.Path(arg).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {path}")
    return str(path)


def parse_kernel_size(arg: str) -> tuple[int, ...]:
    raw = str(arg).strip().lower()
    if not raw:
        raise argparse.ArgumentTypeError("Kernel size string must be non-empty.")
    try:
        return tuple(int(token) for token in raw.split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Kernel size must be formatted like 3x3x3, got {arg!r}"
        ) from exc


def default_scannetpp_local_root() -> str:
    return str(
        (
            pathlib.Path(__file__).resolve().parents[2]
            / "dataset"
            / "scannetpp"
            / "dataset"
            / "data"
        ).resolve()
    )


def make_parser(
    description: str,
    *,
    formatter: type[argparse.HelpFormatter] = argparse.ArgumentDefaultsHelpFormatter,
) -> argparse.ArgumentParser:
    """
    Create a parser with a consistent formatter that shows defaults automatically.
    """
    return argparse.ArgumentParser(
        description=description,
        formatter_class=formatter,
    )


def _dedent(s: str) -> str:
    return textwrap.dedent(s).strip("\n")


# -------------------------
# Reusable argument blocks
# -------------------------


def add_model_selection_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Model selection")
    g.add_argument(
        "--feedforward_model",
        type=str,
        default="da3",
        choices=["da3"],
        help="Feedforward model to use",
    )
    g.add_argument(
        "--video_model",
        type=str,
        default="wan",
        choices=["wan"],
        help="Video model to use",
    )
    g.add_argument(
        "--da3_model_path",
        type=str,
        default=None,
        help="Optional local DA3 model directory containing config.json and model.safetensors.",
    )
    g.add_argument(
        "--da3_config_name",
        type=str,
        default="config.json",
        help="DA3 config filename under --da3_model_path.",
    )
    g.add_argument(
        "--da3_checkpoint_name",
        type=str,
        default="model.safetensors",
        help="DA3 checkpoint filename under --da3_model_path.",
    )
    g.add_argument(
        "--wan_vae_path",
        type=str,
        default=None,
        help=(
            "Optional local Wan VAE directory. If omitted, falls back to auto-detected "
            "local checkpoints or remote repo."
        ),
    )
    g.add_argument(
        "--hf_endpoint",
        type=str,
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Hugging Face endpoint used when local model directories are unavailable.",
    )
    g.add_argument(
        "--wan_vae_repo_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Remote Wan VAE repo id used when --wan_vae_path is not provided.",
    )
    g.add_argument(
        "--da3_ref_view_strategy",
        type=str,
        default="first",
        choices=["saddle_balanced", "first", "middle", "last"],
        help="Reference-view selection strategy used by DA3 anyview backbone.",
    )
    g.add_argument(
        "--da3_enable_scene_flow_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach the pretrained DA3 scene-flow head when loading the anyview model.",
    )
    g.add_argument(
        "--da3_scene_flow_head_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for the DA3 scene-flow head when --da3_enable_scene_flow_head is enabled.",
    )
    return parser


def add_run_and_logging_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Run & logging")
    g.add_argument("--global_seed", type=int, default=23, help="Global seed")
    g.add_argument(
        "--exp_name", type=str, default="wan_da3_stitching", help="Experiment name"
    )

    # Correct boolean handling:
    # - exposes: --wandb_logging / --no-wandb_logging
    g.add_argument(
        "--wandb_logging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable wandb logging",
    )
    g.add_argument(
        "--wandb_project_name",
        type=str,
        default="plan4d",
        help="Wandb project name",
    )
    g.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Log training loss to console every N steps; the first and last step are always logged.",
    )
    return parser


def add_training_loop_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Training loop")
    g.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    g.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default=None,
        help="Path to resume checkpoint",
    )
    g.add_argument(
        "--resume_checkpoint_weights_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, treat --resume_checkpoint_path as a stitched-weights init "
            "source: load stitched_model_step_*.pth or stitched_model_epoch_*.pth and "
            "inherit epoch/global_step/step_in_epoch from resume_checkpoint.pth without "
            "restoring optimizer, scheduler, or DeepSpeed state. Useful when world size changes."
        ),
    )
    g.add_argument(
        "--save_path",
        type=str,
        default="trained_checkpoint/wan_da3_stitching",
        help="Path to save checkpoints",
    )
    g.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="Save a checkpoint every N training steps (matching the logged Epoch Step count). Defaults to 100 when not set.",
    )
    g.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility alias. If provided, converts the old epoch interval "
            "into a training-step interval using the current dataloader steps-per-epoch."
        ),
    )
    g.add_argument(
        "--training_stage",
        type=str,
        default="distill",
        choices=["distill", "gt", "hybrid"],
        help="Training stage for stitching. distill=teacher supervision, gt=Point Odyssey GT geometry, hybrid=randomly interleaved gt+distill steps per epoch.",
    )
    g.add_argument(
        "--use_accelerate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Hugging Face Accelerate for multi-GPU / FSDP training.",
    )
    g.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps when using Accelerate.",
    )
    g.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode for Accelerate training.",
    )
    g.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="DataLoader worker count.",
    )
    return parser


def add_optimizer_args(
    parser: argparse.ArgumentParser,
    *,
    include_warmup: bool = True,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Optimizer")
    g.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    g.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    if include_warmup:
        g.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")

    return parser


def add_common_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Shared data args used by both training and eval.
    Note: dataset is defined here as append; eval can enforce required-ness after parsing if desired,
    but we keep the original behavior for eval by defining it in the eval function instead.
    """
    g = parser.add_argument_group("Data (common)")
    g.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    g.add_argument(
        "--feedforward_resolution",
        type=int,
        default=448,
        help="Image resolution for feedforward model",
    )
    g.add_argument(
        "--da3_process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "square_center_crop"],
        help=(
            "DA3 stitched preprocessing semantics. Also applies to --train_manifest "
            "datasets; default upper_bound_resize keeps long-side upper-bound resize semantics."
        ),
    )
    # These two resolutions can differ because DA3 and Wan VAE use different input sizes.
    return parser


def add_training_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Training-only data arguments.
    """
    add_common_data_args(parser)

    g = parser.add_argument_group("Data (training)")
    g.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    g.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        metavar="NAME:ROOT",
        help=_dedent(
            """
            Provide one or several datasets as NAME:ROOT.
            Example:
              --dataset dl3dv:/data/dl3dv \
              --dataset other:/mnt/other_ds
            """
        ),
    )
    g.add_argument(
        "--train_manifest",
        action="append",
        type=parse_manifest_spec,
        metavar="NAME:PATH",
        help=_dedent(
            """
            Provide one or several manifest jsonl files as NAME:PATH.
            Manifest samples follow --da3_process_res_method; default is upper_bound_resize.
            Example:
              --train_manifest point_odyssey:/path/to/point_odyssey_train.jsonl \
              --train_manifest blinkvision:/path/to/blinkvision_train.jsonl
            """
        ),
    )
    g.add_argument(
        "--manifest_data_root",
        type=str,
        default=None,
        help="Common root used to resolve relative scene_relpath entries stored inside train manifests.",
    )
    g.add_argument(
        "--samples_per_manifest",
        type=int,
        default=None,
        help=(
            "Optional manifest overfit limit. When set, each manifest contributes at most "
            "N samples after stage filtering; scannetpp_nvs_sem_train/val can switch to "
            "local fallback sampling via --scannetpp_local_root."
        ),
    )
    g.add_argument(
        "--scannetpp_local_root",
        type=str,
        default=default_scannetpp_local_root(),
        help=(
            "Local ScanNet++ scene root used when --samples_per_manifest is set and "
            "scannetpp_nvs_sem_train/val manifests need fallback sampling."
        ),
    )
    g.add_argument(
        "--action100m_manifest",
        type=parse_existing_file,
        default=None,
        help=(
            "Optional Action100M video manifest used as an extra distill-only source. "
            "Applies to distill and hybrid training stages."
        ),
    )
    g.add_argument(
        "--num_frames_per_unit_scene",
        type=int,
        default=13,
        help=(
            "Many frames in a sequence will be divided into units of this number of frames, "
            "and each unit will be treated as a separate scene during training"
        ),
    )
    g.add_argument(
        "--num_images_from_unit_scene",
        type=int,
        default=13,
        help="Number of images to sample from each unit scene during training",
    )
    g.add_argument(
        "--point_odyssey_splits",
        type=str,
        default="train",
        help="Comma-separated Point Odyssey splits to include, e.g. train or train,test.",
    )
    return parser




def add_stitching_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Stitching")
    g.add_argument(
        "--stitching_layer_location",
        type=str,
        default="enc_blocks_2",
        help="Location of the stitching layer in the feedforward model",
    )
    g.add_argument(
        "--initialization_weight_path",
        type=str,
        default=None,
        help="Path to the initialization weight for the stitching layer",
    )
    g.add_argument(
        "--init_stitched_model_path",
        type=str,
        default=None,
        help=(
            "Path to a stitched_model_step_*.pth or stitched_model_epoch_*.pth "
            "lightweight checkpoint used only to initialize the stitching layer "
            "and stitched parts without restoring optimizer/scheduler state."
        ),
    )
    g.add_argument(
        "--stitching_layer_config",
        type=parse_conv_spec,
        default=DEFAULT_DA3_STITCHING_SPEC,
        metavar="CONV_SPEC",
        help="See --help in training script for grammar.",
    )
    g.add_argument(
        "--stitching_module_type",
        type=str,
        default="conv3d",
        choices=[
            "conv3d",
            "res_bottleneck",
            "conv_affine",
            "kernel_mixture",
            "axial_attention",
        ],
        help="Stitching module family used between VAE latents and the 3D backbone.",
    )
    g.add_argument(
        "--stitching_hidden_channels",
        type=int,
        default=None,
        help=(
            "Hidden channel width used by non-linear stitching modules. "
            "Defaults to max(in_channels, out_channels) when omitted."
        ),
    )
    g.add_argument(
        "--stitching_norm",
        type=str,
        default="rms",
        choices=["none", "rms", "group", "affine"],
        help="Normalization/calibration type used inside non-linear stitching modules.",
    )
    g.add_argument(
        "--stitching_aux_kernel",
        type=parse_kernel_size,
        default=None,
        help="Auxiliary kernel size used by kernel_mixture, formatted like 3x3x3.",
    )
    g.add_argument(
        "--stitching_attention_heads",
        type=int,
        default=4,
        help="Attention head count used by axial_attention stitching modules.",
    )
    # parse_conv_spec converts this string into a ConvSpec object with kernel/stride/padding.
    g.add_argument(
        "--lora_config",
        type=str,
        default="r8,a16,d0.05,f0",
        help=_dedent(
            """
            LoRA config.
            r<rank>,a<alpha>,d<dropout>,b<bias>,t<targets>,f<0/1>
            Examples:
              r4,a16,d0.05
              r8,a32,tq_proj|k_proj|v_proj
            """
        ),
    )
    g.add_argument(
        "--full_finetune_stitched_parts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, stitched parts stop using LoRA and are trained with full "
            "parameters together with the stitching layer."
        ),
    )
    g.add_argument(
        "--student_da3_init_mode",
        type=str,
        default="pretrained",
        choices=["pretrained", "random"],
        help=(
            "DA3 student initialization mode. pretrained reuses DA3 base weights; "
            "random instantiates the DA3 student architecture without loading base or "
            "scene-flow checkpoints."
        ),
    )
    g.add_argument(
        "--student_train_scope",
        type=str,
        default="default",
        choices=["default", "full_model", "transfer_only", "lora"],
        help=(
            "Trainable scope for the stitched student. default preserves the existing "
            "full_finetune_stitched_parts / LoRA behavior."
        ),
    )
    return parser


def add_geometry_loss_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Geometry loss")
    g.add_argument(
        "--geo_loss_weight",
        type=float,
        default=1.0,
        help="Global weight applied to the geometry loss after component aggregation.",
    )
    g.add_argument(
        "--geo_loss_weight_depth",
        type=float,
        default=1.0,
        help="Depth weight in GT geometry supervision mode.",
    )
    g.add_argument(
        "--geo_loss_weight_ray",
        type=float,
        default=0.01,
        help="Ray weight in GT geometry supervision mode.",
    )
    g.add_argument(
        "--geo_loss_weight_depth_grad",
        type=float,
        default=0.1,
        help="Depth-gradient weight in GT geometry supervision mode.",
    )
    g.add_argument(
        "--geo_loss_weight_point_cloud",
        type=float,
        default=0.01,
        help="Point-cloud weight in GT geometry supervision mode.",
    )
    g.add_argument(
        "--geo_loss_weight_camera",
        type=float,
        default=1.0,
        help="Decoded-camera weight in GT geometry supervision mode.",
    )
    g.add_argument(
        "--geo_loss_weight_temporal",
        type=float,
        default=1.0,
        help="Temporal depth weight in GT geometry supervision mode.",
    )
    g.add_argument(
        "--geo_loss_weight_scene_flow",
        type=float,
        default=1.0,
        help="Scene-flow weight in GT geometry supervision mode when the DA3 scene-flow head is enabled.",
    )
    g.add_argument(
        "--distill_loss_weight_depth",
        type=float,
        default=1.0,
        help="Depth weight in distill supervision mode.",
    )
    g.add_argument(
        "--distill_loss_weight_depth_grad",
        type=float,
        default=1.0,
        help="Depth-gradient weight in distill supervision mode.",
    )
    g.add_argument(
        "--distill_loss_weight_ray",
        type=float,
        default=1.0,
        help="Ray weight in distill supervision mode.",
    )
    g.add_argument(
        "--distill_loss_weight_depth_conf",
        type=float,
        default=0.005,
        help="Depth-confidence weight in distill supervision mode.",
    )
    g.add_argument(
        "--distill_loss_weight_ray_conf",
        type=float,
        default=0.005,
        help="Ray-confidence weight in distill supervision mode.",
    )
    g.add_argument(
        "--distill_loss_weight_camera",
        type=float,
        default=1.0,
        help="Decoded-camera weight in distill supervision mode.",
    )
    g.add_argument(
        "--distill_loss_weight_feature",
        type=float,
        default=0.0,
        help="Feature-token distill weight in distill supervision mode.",
    )
    g.add_argument(
        "--distill_loss_weight_scene_flow",
        type=float,
        default=1.0,
        help="Scene-flow weight in distill supervision mode when the DA3 scene-flow head is enabled.",
    )
    g.add_argument(
        "--geo_loss_warmup_start",
        type=float,
        default=1.0,
        help="Initial geometry-loss warmup scale before ramping to 1.0.",
    )
    g.add_argument(
        "--geo_loss_warmup_steps",
        type=int,
        default=0,
        help="Linear warmup steps for the geometry-loss scale. <=0 disables warmup.",
    )
    return parser


# -------------------------
# Parser builders
# -------------------------


def stitching_training_argument() -> argparse.ArgumentParser:
    parser = make_parser("Stitching training argument")
    add_model_selection_args(parser)
    add_run_and_logging_args(parser)
    add_training_loop_args(parser)
    add_optimizer_args(parser, include_warmup=True)
    add_training_data_args(parser)
    add_stitching_args(parser)
    add_geometry_loss_args(parser)
    return parser


def find_layer_stitching_argument() -> argparse.ArgumentParser:
    parser = make_parser("Find layer for stitching argument")

    g = parser.add_argument_group("Feature extraction")
    g.add_argument(
        "--feature_save_path",
        type=str,
        required=True,
        help="Path to save features used for searching stitching layer",
    )
    g.add_argument(
        "--feature_load_path",
        type=parse_existing_path,
        default=None,
        help=(
            "Optional existing feature cache directory or features.pt file. "
            "When provided, skip extraction and continue fitting from cached features."
        ),
    )
    g.add_argument(
        "--iterations_for_feature_extraction",
        type=int,
        default=100,
        help=(
            "Number of iterations for feature extraction. "
            "Total data used = batch_size × iterations_for_feature_extraction"
        ),
    )

    add_model_selection_args(parser)
    add_training_data_args(parser)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker count used for feature extraction.",
    )
    search_group = parser.add_argument_group("Feature search source")
    search_group.add_argument(
        "--search_input_space",
        type=str,
        default="vae_latent",
        choices=["vae_latent", "vae_decoder"],
        help=(
            "Feature source used to fit transfer Conv3D layers. "
            "vae_latent preserves the original search; vae_decoder decodes cached "
            "VAE latents to intermediate Wan VAE decoder layers before fitting."
        ),
    )
    search_group.add_argument(
        "--vae_decoder_layer_names",
        type=str,
        default=None,
        help=(
            "Comma-separated VAE decode-path module names to search, e.g. "
            "post_quant_conv,decoder.conv_in,decoder.up_blocks.0. "
            "If omitted, a block-level decoder layer list is used."
        ),
    )
    search_group.add_argument(
        "--vae_decoder_layer_mode",
        type=str,
        default="block",
        choices=["block", "all"],
        help=(
            "Decoder layer discovery mode when --vae_decoder_layer_names is omitted. "
            "block searches major Wan decoder blocks; all records every decode-path "
            "module that emits a 5D feature tensor in a dry run."
        ),
    )
    search_group.add_argument(
        "--vae_decoder_max_layers",
        type=int,
        default=None,
        help="Optional cap on discovered VAE decoder source layers for smoke runs.",
    )
    search_group.add_argument(
        "--vae_latent_space",
        type=str,
        default="raw",
        choices=["raw", "normalized"],
        help=(
            "Interpretation of cached VAE latents. raw means encoder samples that can "
            "be passed directly to vae.decode(); normalized means Wan pipeline latents "
            "that must be denormalized before vae.decode()."
        ),
    )
    search_group.add_argument(
        "--decoder_search_device",
        type=str,
        default="cuda",
        help="Device used for VAE decoder feature extraction and Conv3D fitting in vae_decoder mode.",
    )
    search_group.add_argument(
        "--da3_search_layer_count",
        type=int,
        default=None,
        help=(
            "Optional cap on candidate DA3 target layers, preserving cache order. "
            "Use 6 to search only the first six DA3 layers."
        ),
    )
    search_group.add_argument(
        "--da3_search_layer_index",
        type=int,
        default=None,
        help=(
            "When set, only search the DA3 layer at this zero-based index "
            "(within the first --da3_search_layer_count layers). "
            "Used by the launcher to iterate one DA3 layer per round."
        ),
    )
    search_group.add_argument(
        "--vae_decoder_layer_shard_count",
        type=int,
        default=1,
        help="Total number of VAE decoder layer shards for launcher-level parallel search.",
    )
    search_group.add_argument(
        "--vae_decoder_layer_shard_index",
        type=int,
        default=0,
        help="Zero-based VAE decoder layer shard index handled by this process.",
    )
    add_stitching_args(parser)
    return parser










# -------------------------
# Optional post-parse validators
# -------------------------




def apply_feedforward_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply model-specific defaults after parsing.

    DA3 stitching in this repo is currently wired for Point Odyssey training with
    49-frame chunks and a 1536-channel stitching target.
    """
    if getattr(args, "feedforward_model", None) != "da3":
        return args

    if hasattr(args, "training_stage") and getattr(args, "training_stage", None) is None:
        args.training_stage = "distill"

    manifest_mode = bool(getattr(args, "train_manifest", None))
    default_num_frames = 81 if manifest_mode else 49

    if hasattr(args, "num_frames_per_unit_scene"):
        args.num_frames_per_unit_scene = default_num_frames
    if hasattr(args, "num_images_from_unit_scene"):
        args.num_images_from_unit_scene = default_num_frames
    if getattr(args, "resolution", None) == 512:
        args.resolution = 504 if manifest_mode else 532
    if getattr(args, "feedforward_resolution", None) == 448:
        args.feedforward_resolution = 504 if manifest_mode else 532
    if getattr(args, "da3_process_res_method", None) is None:
        args.da3_process_res_method = "upper_bound_resize"

    current_spec = getattr(args, "stitching_layer_config", None)
    if current_spec is None or getattr(current_spec, "out_channels", None) != 1536:
        args.stitching_layer_config = parse_conv_spec(DEFAULT_DA3_STITCHING_SPEC)

    return args
