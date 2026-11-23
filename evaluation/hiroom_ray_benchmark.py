from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_ROOT = PLAN4D_ROOT / "Depth-Anything-3"
DA3_SRC_ROOT = DA3_ROOT / "src"

for import_path in (PROJECT_ROOT, PLAN4D_ROOT, DA3_SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import cv2
import numpy as np
import torch

from evaluation.eth3d_ray_benchmark import (
    DEFAULT_INPUT_RESOLUTION,
    DEFAULT_MAX_FRAMES,
    DEFAULT_STITCHED_INPUT_RESOLUTION,
    _apply_depth_scale_shift_alignment,
    _fit_sequence_depth_scale_shift_alignment,
    _to_float_dict,
    _write_gt_meta,
    _write_raw_camera_params_npz,
    _write_results_npz,
    clone_prediction_dict,
    compute_pose_with_gt,
    compute_raw_camera_scale_metrics,
    default_wan_vae_path,
    dump_json,
    load_anyview_api,
    load_stitched_runner,
    mean_of_dicts,
    resolve_stitched_eval_max_frames,
    run_da3_anyview_scene,
    run_da3_stitched_scene,
    sample_scene_data,
)


def default_hiroom_root() -> str:
    return str((PROJECT_ROOT / "datasets" / "DA3-bench" / "hiroom").resolve())


def default_da3_base_model_path() -> str:
    return str((PLAN4D_ROOT / "checkpoints" / "da3").resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HiRoom full benchmark with ray-derived camera parameters for raw DA3 anyview-base "
            "and/or PLAN4D DA3 stitched models."
        )
    )
    parser.add_argument(
        "--model_kind",
        type=str,
        default="both",
        choices=["da3_anyview_base", "da3_stitched", "both"],
        help="Which model path to evaluate.",
    )
    parser.add_argument(
        "--hiroom_root",
        type=str,
        default=default_hiroom_root(),
        help="HiRoom benchmark root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for benchmark-compatible exports and metric JSON files.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Stitched checkpoint path. Required when model_kind includes da3_stitched.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum frames per HiRoom scene. Uses the benchmark's fixed seed sampling.",
    )
    parser.add_argument(
        "--input_resolution",
        type=int,
        default=None,
        help="Input resolution. Defaults to 532 for anyview-base and 504 for stitched.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of HiRoom scene names.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--da3_model_path",
        type=str,
        default=default_da3_base_model_path(),
        help="Local DA3 anyview weights directory.",
    )
    parser.add_argument(
        "--wan_vae_path",
        type=str,
        default=default_wan_vae_path(),
        help="Local Wan VAE directory for stitched inference.",
    )
    parser.add_argument(
        "--da3_config_name",
        type=str,
        default="config.json",
        help="DA3 config filename under --da3_model_path.",
    )
    parser.add_argument(
        "--da3_checkpoint_name",
        type=str,
        default="model.safetensors",
        help="DA3 checkpoint filename under --da3_model_path.",
    )
    parser.add_argument(
        "--da3_ref_view_strategy",
        type=str,
        default="first",
        choices=["saddle_balanced", "first", "middle", "last"],
        help="Reference-view strategy shared by raw anyview and stitched DA3.",
    )
    parser.add_argument(
        "--stitching_layer_location",
        type=str,
        default="enc_blocks_1",
        help="Fallback stitched layer location when checkpoint metadata is absent.",
    )
    parser.add_argument(
        "--stitching_layer_config",
        type=str,
        default="conv3d_k5x3x3_o1536_s1x2x2_p2x1x1",
        help="Fallback stitched layer spec when checkpoint metadata is absent.",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="r512,a256,d0.05,f0",
        help="Fallback stitched LoRA config when checkpoint metadata is absent.",
    )
    parser.add_argument(
        "--full_finetune_stitched_parts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set when the stitched checkpoint stores fully fine-tuned stitched parts.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HF endpoint passed through to stitched loader when local paths are missing.",
    )
    parser.add_argument(
        "--da3_process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "square_center_crop"],
        help="Preprocess semantics for stitched DA3 inference.",
    )
    parser.add_argument(
        "--stitched_da3_sky_postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply official DA3 mono sky postprocess on HiRoom stitched predictions.",
    )
    parser.add_argument(
        "--stitched_da3_posed_rerun_with_gt_camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, posed evaluation reruns stitched inference with GT camera tokens.",
    )
    parser.add_argument(
        "--hiroom_depth_scale_shift_alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, fit one global affine depth transform d_gt = scale * d_pred + shift "
            "per HiRoom scene before reconstruction fusion."
        ),
    )
    parser.add_argument(
        "--stitched_da3_camera_param_source",
        type=str,
        default="ray_map",
        choices=["ray_map", "camera_token"],
        help=(
            "Which stitched DA3 output to decode camera parameters from. "
            "'ray_map' preserves the default benchmark behavior; 'camera_token' uses cam_dec outputs."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.hiroom_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"HiRoom root does not exist: {dataset_root}")
    if args.input_resolution is not None and args.input_resolution <= 0:
        raise ValueError("--input_resolution must be positive.")
    if args.max_frames == 0:
        raise ValueError("--max_frames must be positive or -1 to disable sampling.")
    if args.model_kind in {"da3_stitched", "both"} and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for da3_stitched or both.")


def list_hiroom_scenes(dataset_root: str | Path) -> list[str]:
    dataset_root = Path(dataset_root)
    scene_list_path = dataset_root / "selected_scene_list_val.txt"
    if scene_list_path.exists():
        scenes = [
            line.strip()
            for line in scene_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if scenes:
            return scenes

    data_root = dataset_root / "data"
    if not data_root.exists():
        return []
    scenes = []
    for path in data_root.glob("*/*/*"):
        if path.is_dir():
            scenes.append(str(path.relative_to(data_root)))
    return sorted(scenes)


def write_hiroom_scene_exports(
    work_dir: str | Path,
    scene_name: str,
    unposed_prediction: dict[str, np.ndarray],
    posed_prediction: dict[str, np.ndarray],
    scene_data,
) -> None:
    base_dir = Path(work_dir) / "model_results" / "hiroom" / scene_name
    for suffix, prediction in (("unposed", unposed_prediction), ("posed", posed_prediction)):
        export_dir = base_dir / suffix
        _write_results_npz(export_dir, prediction)
        _write_raw_camera_params_npz(export_dir, prediction)
        _write_gt_meta(export_dir, scene_data)



def _load_hiroom_gt_depth_and_valid_mask(
    gt_depth_path: str | Path,
    aliasing_mask_path: str | Path,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    target_height, target_width = target_size
    gt_depth_raw = cv2.imread(str(gt_depth_path), -1)
    if gt_depth_raw is None:
        return (
            np.zeros((target_height, target_width), dtype=np.float32),
            np.zeros((target_height, target_width), dtype=bool),
        )
    if gt_depth_raw.ndim == 3:
        gt_depth_raw = gt_depth_raw[..., 0]
    gt_depth = gt_depth_raw.astype(np.float32) / 65535.0 * 100.0
    if gt_depth.shape[:2] != (target_height, target_width):
        gt_depth = cv2.resize(gt_depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    valid_mask = np.isfinite(gt_depth) & (gt_depth > 0.0)
    aliasing_mask = cv2.imread(str(aliasing_mask_path), -1)
    if aliasing_mask is not None:
        if aliasing_mask.ndim == 3:
            aliasing_mask = aliasing_mask[..., 0]
        if aliasing_mask.shape[:2] != (target_height, target_width):
            aliasing_mask = cv2.resize(
                aliasing_mask,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
        valid_mask &= np.logical_not(aliasing_mask > 0)

    gt_depth = np.where(valid_mask, gt_depth, 0.0).astype(np.float32)
    return gt_depth, valid_mask


def _resolve_hiroom_fusion_gt_data(dataset, scene: str, result_path: str | Path):
    full_gt_data = dataset.get_data(scene)
    gt_meta = dataset._load_gt_meta(str(result_path)) if hasattr(dataset, "_load_gt_meta") else None
    if gt_meta is None:
        return full_gt_data, full_gt_data, list(range(len(full_gt_data.image_files)))

    full_image_files = [str(path) for path in full_gt_data.image_files]
    full_index_by_file = {image_file: index for index, image_file in enumerate(full_image_files)}
    image_indices: list[int] = []
    missing_files: list[str] = []
    for image_file in [str(path) for path in gt_meta.image_files]:
        if image_file not in full_index_by_file:
            missing_files.append(image_file)
        else:
            image_indices.append(full_index_by_file[image_file])
    if missing_files:
        raise ValueError(f"HiRoom saved gt_meta references images not present in full scene: {missing_files[:3]}")
    return full_gt_data, gt_meta, image_indices


def _fit_hiroom_depth_scale_shift_alignment(
    pred_depths: np.ndarray,
    full_gt_data,
    image_indices: list[int],
    orig_sizes: list[tuple[int, int]],
) -> tuple[float, float, int]:
    resized_pred_depths: list[np.ndarray] = []
    gt_depths: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    valid_count = 0
    for index in range(len(pred_depths)):
        orig_height, orig_width = orig_sizes[index]
        image_index = image_indices[index]
        pred_depth = cv2.resize(
            pred_depths[index],
            (orig_width, orig_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        gt_depth, valid_mask = _load_hiroom_gt_depth_and_valid_mask(
            full_gt_data.aux.gt_depth_files[image_index],
            full_gt_data.aux.aliasing_mask_files[image_index],
            (orig_height, orig_width),
        )
        valid_mask = valid_mask & np.isfinite(pred_depth) & (pred_depth > 1e-6)
        valid_count += int(np.count_nonzero(valid_mask))
        resized_pred_depths.append(pred_depth)
        gt_depths.append(gt_depth)
        valid_masks.append(valid_mask)

    scale, shift = _fit_sequence_depth_scale_shift_alignment(
        resized_pred_depths,
        gt_depths,
        valid_masks,
    )
    return scale, shift, valid_count


def _prepare_hiroom_fusion_payload(
    dataset,
    scene: str,
    pred_data: SimpleNamespace,
    full_gt_data,
    gt_data,
    image_indices: list[int],
    orig_sizes: list[tuple[int, int]],
    mode: str,
    *,
    depth_scale_shift_alignment: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    if mode == "recon_unposed":
        _, _, umeyama_scale, extrinsics = align_poses_umeyama(
            np.asarray(gt_data.extrinsics, dtype=np.float32).copy(),
            np.asarray(pred_data.extrinsics, dtype=np.float32).copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )
    elif mode == "recon_posed":
        _, _, umeyama_scale, _ = align_poses_umeyama(
            np.asarray(gt_data.extrinsics, dtype=np.float32).copy(),
            np.asarray(pred_data.extrinsics, dtype=np.float32).copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )
        extrinsics = np.stack([full_gt_data.extrinsics[index] for index in image_indices])
    else:
        raise ValueError(f"Invalid mode: {mode}")

    depth_alignment_scale = float(umeyama_scale)
    depth_alignment_shift = 0.0
    use_depth_scale_shift_alignment = bool(depth_scale_shift_alignment)
    if use_depth_scale_shift_alignment:
        depth_alignment_scale, depth_alignment_shift, valid_count = _fit_hiroom_depth_scale_shift_alignment(
            np.asarray(pred_data.depth, dtype=np.float32),
            full_gt_data,
            image_indices,
            orig_sizes,
        )
        if valid_count < 2:
            logger.warning(
                "HiRoom depth scale+shift alignment found no valid GT samples for scene {} mode {}; falling back to Umeyama scale-only.",
                scene,
                mode,
            )
            use_depth_scale_shift_alignment = False
            depth_alignment_scale = float(umeyama_scale)
            depth_alignment_shift = 0.0
        else:
            logger.info(
                "HiRoom depth scale+shift alignment for scene {} mode {}: scale={}, shift={}, valid_pixels={}",
                scene,
                mode,
                depth_alignment_scale,
                depth_alignment_shift,
                valid_count,
            )

    model_height, model_width = pred_data.depth.shape[1], pred_data.depth.shape[2]
    depths_out: list[np.ndarray] = []
    intrinsics_out: list[np.ndarray] = []
    for index in range(len(pred_data.depth)):
        orig_height, orig_width = orig_sizes[index]
        image_index = image_indices[index]
        depth = cv2.resize(
            pred_data.depth[index],
            (orig_width, orig_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        if use_depth_scale_shift_alignment:
            _, valid_mask = _load_hiroom_gt_depth_and_valid_mask(
                full_gt_data.aux.gt_depth_files[image_index],
                full_gt_data.aux.aliasing_mask_files[image_index],
                (orig_height, orig_width),
            )
            valid_mask = valid_mask & np.isfinite(depth) & (depth > 0.0)
            depth = _apply_depth_scale_shift_alignment(
                depth,
                valid_mask,
                scale=depth_alignment_scale,
                shift=depth_alignment_shift,
            )
        else:
            gt_mask = dataset._load_gt_mask(
                full_gt_data.aux.gt_depth_files[image_index],
                full_gt_data.aux.aliasing_mask_files[image_index],
            )
            depth = dataset._mask_invalid_depth(depth, gt_mask) * float(umeyama_scale)
        depths_out.append(depth.astype(np.float32))

        if mode == "recon_unposed":
            height_ratio = orig_height / float(model_height)
            width_ratio = orig_width / float(model_width)
            intrinsics = pred_data.intrinsics[index].copy()
            intrinsics[0, :] *= width_ratio
            intrinsics[1, :] *= height_ratio
        else:
            intrinsics = full_gt_data.intrinsics[image_index].copy()
        intrinsics_out.append(intrinsics.astype(np.float32))

    return np.stack(depths_out), np.stack(intrinsics_out), np.asarray(extrinsics, dtype=np.float32)


def _fuse_hiroom_scene(
    dataset,
    scene: str,
    result_path: str | Path,
    fuse_path: str | Path,
    mode: str,
    *,
    depth_scale_shift_alignment: bool = False,
) -> None:
    if not depth_scale_shift_alignment:
        dataset.fuse3d(scene, str(result_path), str(fuse_path), mode)
        return

    import open3d as o3d
    from depth_anything_3.bench.dataset import _wait_for_file_ready
    from depth_anything_3.bench.utils import (
        create_tsdf_volume,
        fuse_depth_to_tsdf,
        sample_points_from_mesh,
        write_point_cloud_with_cameras,
    )

    result_path = Path(result_path)
    fuse_path = Path(fuse_path)
    full_gt_data, gt_data, image_indices = _resolve_hiroom_fusion_gt_data(dataset, scene, result_path)
    _wait_for_file_ready(str(result_path))
    with np.load(result_path) as pred_npz:
        pred_data = SimpleNamespace(**{key: value for key, value in pred_npz.items()})

    images: list[np.ndarray] = []
    orig_sizes: list[tuple[int, int]] = []
    for image_index in image_indices:
        image_path = full_gt_data.image_files[image_index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read HiRoom image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        orig_sizes.append((image.shape[0], image.shape[1]))
    images_np = np.stack(images, axis=0)

    depths, intrinsics, extrinsics = _prepare_hiroom_fusion_payload(
        dataset,
        scene,
        pred_data,
        full_gt_data,
        gt_data,
        image_indices,
        orig_sizes,
        mode,
        depth_scale_shift_alignment=True,
    )

    volume = create_tsdf_volume(voxel_length=dataset.voxel_length, sdf_trunc=dataset.sdf_trunc)
    mesh = fuse_depth_to_tsdf(
        volume,
        depths,
        images_np,
        intrinsics,
        extrinsics,
        max_depth=dataset.max_depth,
    )
    pcd = sample_points_from_mesh(mesh, dataset.sampling_number)
    fuse_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(fuse_path), pcd)
    write_point_cloud_with_cameras(
        fuse_path=str(fuse_path),
        pcd=pcd,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_sizes=orig_sizes,
    )


def evaluate_hiroom_work_dir(
    work_dir: str | Path,
    dataset,
    scenes: list[str],
    *,
    depth_scale_shift_alignment: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    work_dir = Path(work_dir)
    pose_results: dict[str, dict[str, float]] = {}
    recon_unposed_results: dict[str, dict[str, float]] = {}
    recon_posed_results: dict[str, dict[str, float]] = {}

    for scene in scenes:
        base_dir = work_dir / "model_results" / "hiroom" / scene
        unposed_result_path = base_dir / "unposed" / "exports" / "mini_npz" / "results.npz"
        posed_result_path = base_dir / "posed" / "exports" / "mini_npz" / "results.npz"
        gt_meta = np.load(base_dir / "unposed" / "exports" / "gt_meta.npz", allow_pickle=True)

        pose_results[scene] = compute_pose_with_gt(
            result_path=unposed_result_path,
            gt_extrinsics=gt_meta["extrinsics"],
        )

        unposed_fuse_path = base_dir / "unposed" / "exports" / "fuse" / "pcd.ply"
        _fuse_hiroom_scene(
            dataset,
            scene,
            unposed_result_path,
            unposed_fuse_path,
            "recon_unposed",
            depth_scale_shift_alignment=depth_scale_shift_alignment,
        )
        recon_unposed_results[scene] = _to_float_dict(dataset.eval3d(scene, str(unposed_fuse_path)))

        posed_fuse_path = base_dir / "posed" / "exports" / "fuse" / "pcd.ply"
        _fuse_hiroom_scene(
            dataset,
            scene,
            posed_result_path,
            posed_fuse_path,
            "recon_posed",
            depth_scale_shift_alignment=depth_scale_shift_alignment,
        )
        recon_posed_results[scene] = _to_float_dict(dataset.eval3d(scene, str(posed_fuse_path)))

    pose_results["mean"] = mean_of_dicts(list(pose_results.values()))
    recon_unposed_results["mean"] = mean_of_dicts(list(recon_unposed_results.values()))
    recon_posed_results["mean"] = mean_of_dicts(list(recon_posed_results.values()))

    metric_dir = work_dir / "metric_results"
    dump_json(metric_dir / "hiroom_pose.json", pose_results)
    dump_json(metric_dir / "hiroom_recon_unposed.json", recon_unposed_results)
    dump_json(metric_dir / "hiroom_recon_posed.json", recon_posed_results)

    return {
        "pose": pose_results,
        "recon_unposed": recon_unposed_results,
        "recon_posed": recon_posed_results,
    }


def evaluate_hiroom_raw_camera_diagnostics_work_dir(
    work_dir: str | Path,
    scenes: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    work_dir = Path(work_dir)
    raw_camera_pose_unposed: dict[str, dict[str, float]] = {}
    raw_camera_pose_posed: dict[str, dict[str, float]] = {}
    raw_camera_scale_unposed: dict[str, dict[str, float]] = {}
    raw_camera_scale_posed: dict[str, dict[str, float]] = {}

    for scene in scenes:
        base_dir = work_dir / "model_results" / "hiroom" / scene
        gt_meta = np.load(base_dir / "unposed" / "exports" / "gt_meta.npz", allow_pickle=True)
        gt_extrinsics = np.asarray(gt_meta["extrinsics"], dtype=np.float32)

        unposed_raw_path = base_dir / "unposed" / "exports" / "raw_camera_params.npz"
        posed_raw_path = base_dir / "posed" / "exports" / "raw_camera_params.npz"

        raw_camera_pose_unposed[scene] = compute_pose_with_gt(
            result_path=unposed_raw_path,
            gt_extrinsics=gt_extrinsics,
        )
        raw_camera_pose_posed[scene] = compute_pose_with_gt(
            result_path=posed_raw_path,
            gt_extrinsics=gt_extrinsics,
        )

        unposed_raw = np.load(unposed_raw_path)
        posed_raw = np.load(posed_raw_path)
        raw_camera_scale_unposed[scene] = compute_raw_camera_scale_metrics(
            unposed_raw["extrinsics"],
            gt_extrinsics,
        )
        raw_camera_scale_posed[scene] = compute_raw_camera_scale_metrics(
            posed_raw["extrinsics"],
            gt_extrinsics,
        )

    raw_camera_pose_unposed["mean"] = mean_of_dicts(list(raw_camera_pose_unposed.values()))
    raw_camera_pose_posed["mean"] = mean_of_dicts(list(raw_camera_pose_posed.values()))
    raw_camera_scale_unposed["mean"] = mean_of_dicts(list(raw_camera_scale_unposed.values()))
    raw_camera_scale_posed["mean"] = mean_of_dicts(list(raw_camera_scale_posed.values()))

    metric_dir = work_dir / "metric_results"
    dump_json(metric_dir / "hiroom_raw_camera_pose_unposed.json", raw_camera_pose_unposed)
    dump_json(metric_dir / "hiroom_raw_camera_pose_posed.json", raw_camera_pose_posed)
    dump_json(metric_dir / "hiroom_raw_camera_scale_unposed.json", raw_camera_scale_unposed)
    dump_json(metric_dir / "hiroom_raw_camera_scale_posed.json", raw_camera_scale_posed)

    return {
        "raw_camera_pose_unposed": raw_camera_pose_unposed,
        "raw_camera_pose_posed": raw_camera_pose_posed,
        "raw_camera_scale_unposed": raw_camera_scale_unposed,
        "raw_camera_scale_posed": raw_camera_scale_posed,
    }


def _configure_hiroom_dataset(dataset, hiroom_root: str | Path) -> None:
    hiroom_root = Path(hiroom_root)
    dataset.data_root = str((hiroom_root / "data").resolve())
    dataset.gt_root_path = str((hiroom_root / "fused_pcd").resolve())


def evaluate_model_kind(
    args: argparse.Namespace,
    *,
    model_kind: str,
    scenes: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    from depth_anything_3.bench.datasets.hiroom import HiRoomDataset

    output_root = Path(args.output_dir) / model_kind
    output_root.mkdir(parents=True, exist_ok=True)

    effective_args = SimpleNamespace(**vars(args))
    if effective_args.input_resolution is None:
        if model_kind == "da3_stitched":
            effective_args.input_resolution = DEFAULT_STITCHED_INPUT_RESOLUTION
        else:
            effective_args.input_resolution = DEFAULT_INPUT_RESOLUTION

    dataset = HiRoomDataset()
    _configure_hiroom_dataset(dataset, effective_args.hiroom_root)

    if model_kind == "da3_anyview_base":
        runner = load_anyview_api(effective_args)

        def run_scene(scene_data):
            return (
                run_da3_anyview_scene(
                    runner,
                    scene_data=scene_data,
                    process_res=effective_args.input_resolution,
                    ref_view_strategy=effective_args.da3_ref_view_strategy,
                ),
                run_da3_anyview_scene(
                    runner,
                    scene_data=scene_data,
                    process_res=effective_args.input_resolution,
                    ref_view_strategy=effective_args.da3_ref_view_strategy,
                    extrinsics=scene_data.extrinsics,
                    intrinsics=scene_data.intrinsics,
                ),
            )
    else:
        runner = load_stitched_runner(
            effective_args,
            da3_process_res_method=getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
        )

        def run_scene(scene_data):
            unposed_prediction, _ = run_da3_stitched_scene(
                runner,
                scene_data=scene_data,
                args=effective_args,
                process_res_method=getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
            )
            if getattr(effective_args, "stitched_da3_posed_rerun_with_gt_camera", True):
                posed_prediction, _ = run_da3_stitched_scene(
                    runner,
                    scene_data=scene_data,
                    args=effective_args,
                    extrinsics=scene_data.extrinsics,
                    intrinsics=scene_data.intrinsics,
                    process_res_method=getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
                )
            else:
                posed_prediction = clone_prediction_dict(unposed_prediction)
            return unposed_prediction, posed_prediction

    for scene in scenes:
        full_scene_data = dataset.get_data(scene)
        scene_max_frames = effective_args.max_frames
        if model_kind == "da3_stitched":
            scene_max_frames = resolve_stitched_eval_max_frames(
                total_frames=len(full_scene_data.image_files),
                requested_max_frames=effective_args.max_frames,
            )
        scene_data = sample_scene_data(full_scene_data, scene_max_frames)
        logger.info(
            "Evaluating {} scene {} with {} frames",
            model_kind,
            scene,
            len(scene_data.image_files),
        )
        unposed_prediction, posed_prediction = run_scene(scene_data)
        write_hiroom_scene_exports(
            work_dir=output_root,
            scene_name=scene,
            unposed_prediction=unposed_prediction,
            posed_prediction=posed_prediction,
            scene_data=scene_data,
        )

    depth_scale_shift_alignment = bool(getattr(effective_args, "hiroom_depth_scale_shift_alignment", False))
    metrics = evaluate_hiroom_work_dir(
        output_root,
        dataset,
        scenes,
        depth_scale_shift_alignment=depth_scale_shift_alignment,
    )
    metrics.update(evaluate_hiroom_raw_camera_diagnostics_work_dir(output_root, scenes))
    dump_json(
        output_root / "run_meta.json",
        {
            "model_kind": model_kind,
            "dataset_name": "hiroom",
            "hiroom_root": str(Path(effective_args.hiroom_root).resolve()),
            "input_resolution": effective_args.input_resolution,
            "da3_ref_view_strategy": effective_args.da3_ref_view_strategy,
            "da3_process_res_method": getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
            "stitched_da3_sky_postprocess": getattr(effective_args, "stitched_da3_sky_postprocess", True),
            "stitched_da3_posed_rerun_with_gt_camera": getattr(
                effective_args,
                "stitched_da3_posed_rerun_with_gt_camera",
                True,
            ),
            "hiroom_depth_scale_shift_alignment": depth_scale_shift_alignment,
            "stitched_da3_camera_param_source": getattr(
                effective_args,
                "stitched_da3_camera_param_source",
                "ray_map",
            ),
            "max_frames": effective_args.max_frames,
            "device": effective_args.device,
            "ray_pose_only": True,
            "force_fp32_inference": True,
            "scenes": scenes,
        },
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    available_scenes = list_hiroom_scenes(args.hiroom_root)
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [scene for scene in available_scenes if scene in wanted]
    else:
        scenes = available_scenes
    if not scenes:
        raise ValueError(f"No HiRoom scenes selected under {args.hiroom_root}")

    model_kinds = ["da3_anyview_base", "da3_stitched"] if args.model_kind == "both" else [args.model_kind]
    for model_kind in model_kinds:
        logger.info("Running HiRoom benchmark for {} on {} scenes", model_kind, len(scenes))
        evaluate_model_kind(args, model_kind=model_kind, scenes=scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
