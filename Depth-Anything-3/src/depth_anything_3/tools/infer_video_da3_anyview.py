from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch

from depth_anything_3.utils.scene_flow_postprocess import (
    filter_scene_flow_periodic,
    scene_flow_filter_period,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "weights"
DEFAULT_EXPORT_FORMAT = "mini_npz-npz-glb"
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def select_frame_indices(
    *,
    total_frames: int,
    source_fps: float,
    sample_fps: float | None,
    frame_stride: int,
    max_frames: int | None,
) -> list[int]:
    if total_frames <= 0:
        raise ValueError(f"Expected positive total_frames, got {total_frames}")

    if sample_fps is not None and sample_fps > 0 and source_fps > 0:
        duration = total_frames / source_fps
        timestamps = np.arange(0.0, duration, 1.0 / sample_fps, dtype=np.float64)
        indices = np.floor(timestamps * source_fps + 1e-6).astype(np.int64)
        selected = sorted({int(index) for index in indices if 0 <= index < total_frames})
    else:
        stride = max(1, int(frame_stride))
        selected = list(range(0, total_frames, stride))

    if not selected:
        selected = [0]

    if max_frames is not None and max_frames > 0 and len(selected) > max_frames:
        keep_positions = np.linspace(0, len(selected) - 1, int(max_frames)).astype(np.int64)
        selected = [selected[int(position)] for position in keep_positions]

    return selected


def write_ascii_point_cloud_ply(
    output_path: str | Path,
    points: Iterable[Sequence[float]],
    colors: Iterable[Sequence[int]],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points_array = np.asarray(points, dtype=np.float32)
    colors_array = np.asarray(colors, dtype=np.uint8)
    if points_array.size == 0:
        points_array = np.zeros((0, 3), dtype=np.float32)
    if colors_array.size == 0:
        colors_array = np.zeros((0, 3), dtype=np.uint8)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N,3], got {points_array.shape}")
    if colors_array.ndim != 2 or colors_array.shape[1] != 3:
        raise ValueError(f"Expected colors with shape [N,3], got {colors_array.shape}")
    if len(points_array) != len(colors_array):
        raise ValueError("points and colors must have the same length")

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points_array)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points_array, colors_array):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local DA3 any-view inference on an RGB video or image directory with "
            "ray-map camera solving and point-cloud export."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", "--video-path", help="Input RGB video.")
    input_group.add_argument(
        "--image_dir",
        "--image-dir",
        help="Directory containing an RGB image sequence.",
    )
    parser.add_argument("--output_dir", "--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--model_dir",
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="DA3 any-view local weights directory.",
    )
    parser.add_argument(
        "--scene_flow_head_checkpoint_path",
        "--scene-flow-head-checkpoint-path",
        default=None,
        help="Load a trained scene-flow head and export forward 3D displacement.",
    )
    parser.add_argument(
        "--scene_flow_preset",
        "--scene-flow-preset",
        default="da3-giant-sceneflow",
        help="DA3 architecture preset for the scene-flow head and adapter.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--process_res", "--process-res", type=int, default=504)
    parser.add_argument(
        "--process_res_method",
        "--process-res-method",
        default="upper_bound_resize",
        help="DA3 preprocess method, e.g. upper_bound_resize.",
    )
    parser.add_argument(
        "--sample_fps",
        "--sample-fps",
        type=float,
        default=0.0,
        help="Sample video at this FPS. Set <=0 to use --frame_stride.",
    )
    parser.add_argument("--frame_stride", "--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max_frames",
        "--max-frames",
        type=int,
        default=120,
        help="Evenly cap sampled frames. Set <=0 to disable.",
    )
    parser.add_argument(
        "--ref_view_strategy",
        "--ref-view-strategy",
        default="first",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
        help="DA3 reference-view strategy. PLAN4D ETH3D raw anyview defaults to first.",
    )
    parser.add_argument("--export_format", "--export-format", default=DEFAULT_EXPORT_FORMAT)
    parser.add_argument(
        "--conf_thresh_percentile",
        "--conf-thresh-percentile",
        type=float,
        default=40.0,
    )
    parser.add_argument("--num_max_points", "--num-max-points", type=int, default=1_000_000)
    parser.add_argument(
        "--point_cloud_max_points",
        "--point-cloud-max-points",
        type=int,
        default=1_000_000,
    )
    parser.add_argument("--point_cloud_seed", "--point-cloud-seed", type=int, default=42)
    parser.add_argument(
        "--save_point_cloud",
        "--save-point-cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write reconstruction_pcd.ply from predicted depth and ray-solved cameras.",
    )
    parser.add_argument(
        "--save_per_frame_arrays",
        "--save-per-frame-arrays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-frame depth/conf/camera .npy files.",
    )
    parser.add_argument(
        "--show_cameras",
        "--show-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show cameras in DA3 GLB export.",
    )
    parser.add_argument(
        "--autocast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch autocast. Disabled by default to match PLAN4D raw anyview runs.",
    )
    parser.add_argument(
        "--allow_existing_output_dir",
        "--allow-existing-output-dir",
        action="store_true",
        help="Allow writing into an existing non-empty output directory. No cleanup is performed.",
    )
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument(
        "--scene_flow_degrid_period", "--scene-flow-degrid-period",
        type=scene_flow_filter_period, default=0.0, help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def prepare_output_dir(output_dir: Path, *, allow_existing_output_dir: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing_output_dir:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --allow-existing-output-dir or choose a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def read_video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if frame_count <= 0:
        raise ValueError(f"Video reports no frames: {video_path}")
    return {
        "frame_count": frame_count,
        "fps": source_fps,
        "width": width,
        "height": height,
    }


def natural_path_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
        if part
    )


def list_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Input image directory does not exist: {image_dir}")
    image_paths = sorted(
        (
            path.resolve()
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=natural_path_key,
    )
    if not image_paths:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"No supported images found in {image_dir}; expected: {extensions}")
    return image_paths


def read_image_dir_metadata(image_paths: Sequence[Path]) -> dict[str, Any]:
    first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Failed to read image: {image_paths[0]}")
    height, width = first_image.shape[:2]
    return {
        "frame_count": len(image_paths),
        "fps": 0.0,
        "width": int(width),
        "height": int(height),
    }


def select_image_frames(
    image_paths: Sequence[Path],
    frame_indices: Sequence[int],
) -> tuple[list[str], list[dict[str, Any]]]:
    selected_paths: list[str] = []
    frame_records: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(frame_indices):
        image_path = str(image_paths[source_index])
        selected_paths.append(image_path)
        frame_records.append(
            {
                "output_index": output_index,
                "source_frame_index": int(source_index),
                "timestamp_sec": None,
                "path": image_path,
            }
        )
    return selected_paths, frame_records


def extract_video_frames(
    *,
    video_path: Path,
    frame_dir: Path,
    frame_indices: Sequence[int],
    source_fps: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frame_paths: list[str] = []
    frame_records: list[dict[str, Any]] = []
    try:
        for output_index, source_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to read frame {source_index} from {video_path}")

            frame_path = frame_dir / f"{output_index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame_bgr):
                raise RuntimeError(f"Failed to write frame: {frame_path}")

            timestamp_sec = float(source_index / source_fps) if source_fps > 0 else math.nan
            frame_paths.append(str(frame_path))
            frame_records.append(
                {
                    "output_index": output_index,
                    "source_frame_index": int(source_index),
                    "timestamp_sec": timestamp_sec,
                    "path": str(frame_path),
                }
            )
    finally:
        capture.release()

    return frame_paths, frame_records


def intrinsics_to_normalized_camera_params(
    intrinsics: np.ndarray,
    *,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    focal_lengths = np.stack(
        [
            intrinsics[..., 0, 0] * 2.0 / float(image_width),
            intrinsics[..., 1, 1] * 2.0 / float(image_height),
        ],
        axis=-1,
    )
    principal_points = np.stack(
        [
            intrinsics[..., 0, 2] * 2.0 / float(image_width),
            intrinsics[..., 1, 2] * 2.0 / float(image_height),
        ],
        axis=-1,
    )
    return focal_lengths.astype(np.float32), principal_points.astype(np.float32)


def as_homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.zeros(extrinsics.shape[:-2] + (1, 4), dtype=np.float64)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([extrinsics, bottom], axis=-2)
    raise ValueError(f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}")


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(as_homogeneous_extrinsics(extrinsics))
    return np.asarray(c2w[..., :3, 3], dtype=np.float32)


def index_colors(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    values = np.linspace(0, 255, count, dtype=np.uint8)
    return np.stack([values, 255 - values, np.full_like(values, 64)], axis=-1)


def write_prediction_npz(output_dir: Path, prediction: Any) -> None:
    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    image_height = int(prediction.depth.shape[-2])
    image_width = int(prediction.depth.shape[-1])
    focal_lengths, principal_points = intrinsics_to_normalized_camera_params(
        np.asarray(prediction.intrinsics),
        image_height=image_height,
        image_width=image_width,
    )

    np.savez_compressed(
        export_dir / "raw_camera_params.npz",
        extrinsics=np.asarray(prediction.extrinsics, dtype=np.float32),
        intrinsics=np.asarray(prediction.intrinsics, dtype=np.float32),
        focal_lengths=focal_lengths,
        principal_points=principal_points,
    )
    scene_flow = getattr(prediction, "scene_flow", None)
    if scene_flow is not None:
        np.savez_compressed(
            export_dir / "scene_flow.npz",
            scene_flow=np.asarray(scene_flow, dtype=np.float32),
        )


def write_camera_poses_txt(
    output_dir: Path,
    frame_records: Sequence[dict[str, Any]],
    prediction: Any,
) -> None:
    pose_path = output_dir / "camera_poses.txt"
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
    with pose_path.open("w", encoding="utf-8") as file:
        file.write("# camera_source: ray_map(use_ray_pose=True)\n")
        file.write("# columns: output_index source_frame_index timestamp_sec w2c_3x4_row_major\n")
        for record, extrinsic in zip(frame_records, extrinsics):
            matrix = extrinsic[:3, :4].reshape(-1)
            values = " ".join(f"{float(value):.9g}" for value in matrix)
            timestamp = record["timestamp_sec"]
            timestamp_text = (
                "nan" if timestamp is None or math.isnan(timestamp) else f"{timestamp:.9g}"
            )
            file.write(
                f"{record['output_index']} {record['source_frame_index']} "
                f"{timestamp_text} {values}\n"
            )


def write_per_frame_arrays(
    output_dir: Path,
    frame_records: Sequence[dict[str, Any]],
    prediction: Any,
) -> None:
    depth_dir = output_dir / "depth"
    conf_dir = output_dir / "confidence"
    camera_dir = output_dir / "camera"
    depth_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    depth = np.asarray(prediction.depth, dtype=np.float32)
    conf = None if prediction.conf is None else np.asarray(prediction.conf, dtype=np.float32)
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
    for record, depth_frame, extrinsic, intrinsic in zip(
        frame_records,
        depth,
        extrinsics,
        intrinsics,
    ):
        prefix = f"{int(record['output_index']):06d}"
        np.save(depth_dir / f"{prefix}.npy", depth_frame)
        np.save(camera_dir / f"{prefix}_extrinsics_w2c.npy", extrinsic)
        np.save(camera_dir / f"{prefix}_intrinsics.npy", intrinsic)
        if conf is not None:
            np.save(conf_dir / f"{prefix}.npy", conf[int(record["output_index"])])


def write_reconstruction_point_cloud(
    output_dir: Path,
    prediction: Any,
    args: argparse.Namespace,
) -> int:
    from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors

    if prediction.processed_images is None:
        raise ValueError("prediction.processed_images is required for point cloud export")
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise ValueError("prediction extrinsics/intrinsics are required for point cloud export")

    conf_threshold = 0.0
    if prediction.conf is not None:
        conf_threshold = float(np.percentile(prediction.conf, args.conf_thresh_percentile))

    points, colors = _depths_to_world_points_with_colors(
        np.asarray(prediction.depth),
        np.asarray(prediction.intrinsics),
        np.asarray(prediction.extrinsics),
        np.asarray(prediction.processed_images),
        None if prediction.conf is None else np.asarray(prediction.conf),
        conf_threshold,
    )
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]

    max_points = int(args.point_cloud_max_points)
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(int(args.point_cloud_seed))
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    write_ascii_point_cloud_ply(output_dir / "reconstruction_pcd.ply", points, colors)
    camera_centers = camera_centers_from_extrinsics(np.asarray(prediction.extrinsics))
    write_ascii_point_cloud_ply(
        output_dir / "camera_poses.ply",
        camera_centers,
        index_colors(len(camera_centers)),
    )
    return int(len(points))


def run_da3_inference(frame_paths: Sequence[str], args: argparse.Namespace) -> Any:
    from depth_anything_3.api import DepthAnything3

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model_dir)
    infer_scene_flow = args.scene_flow_head_checkpoint_path is not None
    if infer_scene_flow:
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        from utils.utils_for_thirdparty import _load_scene_flow_head_checkpoint

        preset = load_config(MODEL_REGISTRY[args.scene_flow_preset])
        core = model.model.da3 if hasattr(model.model, "da3") else model.model
        core.scene_flow_head = create_object(preset.scene_flow_head)
        core.scene_flow_adapter = create_object(preset.scene_flow_adapter)
        _load_scene_flow_head_checkpoint(
            core.scene_flow_head, Path(args.scene_flow_head_checkpoint_path)
        )
    model = model.to(device=device)
    model.eval()
    prediction = model.inference(
        list(frame_paths),
        process_res=int(args.process_res),
        process_res_method=args.process_res_method,
        export_dir=args.output_dir,
        export_format=args.export_format,
        use_ray_pose=True,
        infer_scene_flow=infer_scene_flow,
        ref_view_strategy=args.ref_view_strategy,
        conf_thresh_percentile=float(args.conf_thresh_percentile),
        num_max_points=int(args.num_max_points),
        show_cameras=bool(args.show_cameras),
        autocast_enabled=bool(args.autocast),
    )
    period = getattr(args, "scene_flow_degrid_period", 0.0)
    if period and getattr(prediction, "scene_flow", None) is not None:
        prediction.scene_flow = filter_scene_flow_periodic(
            prediction.scene_flow, period, channel_dim=-1
        )
    return prediction


def write_summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    input_type: str,
    input_path: Path,
    input_metadata: dict[str, Any],
    frame_records: Sequence[dict[str, Any]],
    point_count: int | None,
) -> None:
    summary = {
        "input_type": input_type,
        "input_path": str(input_path),
        "model_dir": str(Path(args.model_dir).resolve()),
        "camera_source": "ray_map(use_ray_pose=True)",
        "process_res": int(args.process_res),
        "process_res_method": args.process_res_method,
        "ref_view_strategy": args.ref_view_strategy,
        "export_format": args.export_format,
        "scene_flow_head_checkpoint_path": args.scene_flow_head_checkpoint_path,
        "scene_flow_preset": args.scene_flow_preset,
        "num_selected_frames": len(frame_records),
        "frames": list(frame_records),
        "outputs": {
            "mini_npz": "exports/mini_npz/results.npz",
            "npz": "exports/npz/results.npz" if "npz" in args.export_format else None,
            "raw_camera_params": "exports/raw_camera_params.npz",
            "scene_flow": (
                "exports/scene_flow.npz"
                if (output_dir / "exports/scene_flow.npz").is_file()
                else None
            ),
            "camera_poses_txt": "camera_poses.txt",
            "reconstruction_pcd": "reconstruction_pcd.ply" if point_count is not None else None,
            "camera_poses_ply": "camera_poses.ply" if point_count is not None else None,
            "scene_glb": "scene.glb" if "glb" in args.export_format else None,
        },
        "point_count": point_count,
    }
    if getattr(args, "scene_flow_degrid_period", 0.0):
        summary["scene_flow_degrid_period"] = args.scene_flow_degrid_period
    if input_type == "video":
        summary["video_path"] = str(input_path)
        summary["video"] = input_metadata
    else:
        summary["image_dir"] = str(input_path)
        summary["images"] = input_metadata
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(output_dir)
    args.model_dir = str(Path(args.model_dir).expanduser().resolve())
    if int(args.max_frames) <= 0:
        args.max_frames = None
    if float(args.sample_fps) <= 0:
        args.sample_fps = None

    if not Path(args.model_dir).is_dir():
        raise FileNotFoundError(f"DA3 model directory does not exist: {args.model_dir}")
    if args.scene_flow_head_checkpoint_path is not None:
        checkpoint_path = Path(args.scene_flow_head_checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Scene-flow head checkpoint does not exist: {checkpoint_path}")
        args.scene_flow_head_checkpoint_path = str(checkpoint_path)

    image_paths: list[Path] | None = None
    if args.video_path is not None:
        input_type = "video"
        input_path = Path(args.video_path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")
        input_metadata = read_video_metadata(input_path)
    else:
        input_type = "image_dir"
        input_path = Path(args.image_dir).expanduser().resolve()
        if args.sample_fps is not None:
            raise ValueError("--sample_fps is only supported for video input")
        image_paths = list_image_files(input_path)
        input_metadata = read_image_dir_metadata(image_paths)

    prepare_output_dir(output_dir, allow_existing_output_dir=args.allow_existing_output_dir)
    frame_indices = select_frame_indices(
        total_frames=int(input_metadata["frame_count"]),
        source_fps=float(input_metadata["fps"]),
        sample_fps=args.sample_fps,
        frame_stride=int(args.frame_stride),
        max_frames=args.max_frames,
    )

    print(f"[INFO] Input {input_type}: {input_path}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Model dir: {args.model_dir}")
    print(f"[INFO] Selected frames: {len(frame_indices)} / {input_metadata['frame_count']}")
    print("[INFO] Camera source: ray_map(use_ray_pose=True)")
    if args.dry_run:
        return 0

    if input_type == "video":
        frame_paths, frame_records = extract_video_frames(
            video_path=input_path,
            frame_dir=output_dir / "frames",
            frame_indices=frame_indices,
            source_fps=float(input_metadata["fps"]),
        )
    else:
        assert image_paths is not None
        frame_paths, frame_records = select_image_frames(image_paths, frame_indices)
    prediction = run_da3_inference(frame_paths, args)
    write_prediction_npz(output_dir, prediction)
    write_camera_poses_txt(output_dir, frame_records, prediction)
    if args.save_per_frame_arrays:
        write_per_frame_arrays(output_dir, frame_records, prediction)

    point_count = None
    if args.save_point_cloud:
        point_count = write_reconstruction_point_cloud(output_dir, prediction, args)

    write_summary(
        output_dir=output_dir,
        args=args,
        input_type=input_type,
        input_path=input_path,
        input_metadata=input_metadata,
        frame_records=frame_records,
        point_count=point_count,
    )
    print(f"[INFO] Results written to: {output_dir}")
    if point_count is not None:
        print(f"[INFO] PLY points: {point_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
