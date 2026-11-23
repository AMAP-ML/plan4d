from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_SRC_ROOT = PLAN4D_ROOT / "Depth-Anything-3" / "src"

for import_path in (PROJECT_ROOT, PLAN4D_ROOT, DA3_SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from depth_anything_3.utils.geometry import affine_inverse_np, as_homogeneous
from depth_anything_3.utils.pose_align import align_poses_umeyama


DEFAULT_SPLIT = "unposed"
DEFAULT_SKIP_SCENES = ("courtyard",)
DEFAULT_FPS = 6
DEFAULT_MAX_POINT_CLOUD_POINTS = 60000
DEFAULT_STATIC_XYTOP_NAME = "camera_compare_da3_pred_gt_static_xytop.png"
DEFAULT_VIDEO_XYTOP_NAME = "camera_compare_da3_pred_gt_dynamic_xytop.mp4"
DEFAULT_FRAMES_XYTOP_DIRNAME = "camera_compare_da3_pred_gt_frames_xytop"
DEFAULT_META_XYTOP_NAME = "camera_compare_da3_pred_gt_meta_xytop.npz"


def default_da3_root() -> str:
    return str(
        (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "da3_anyview"
            / "model_results"
            / "eth3d"
        ).resolve()
    )


def default_pred_root() -> str:
    return str(
        (
            PROJECT_ROOT
            / "outputs"
            / "evaluation"
            / "da3_stitched"
            / "model_results"
            / "eth3d"
        ).resolve()
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render ETH3D DA3/pred/GT camera trajectories with a 3D perspective view "
            "and an XY-orthographic top view."
        )
    )
    parser.add_argument(
        "--da3_root",
        type=str,
        default=default_da3_root(),
        help="ETH3D scene result root for raw DA3 anyview outputs.",
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default=default_pred_root(),
        help="ETH3D scene result root for stitched prediction outputs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        choices=["unposed", "posed"],
        help="Which subdirectory under each scene to visualize.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional scene subset. Defaults to all shared scenes.",
    )
    parser.add_argument(
        "--skip_scenes",
        nargs="*",
        default=list(DEFAULT_SKIP_SCENES),
        help="Scenes to skip even if data exists. Defaults to already rendered first-part scenes.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate outputs even when the target xytop files already exist.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--max_point_cloud_points",
        type=int,
        default=DEFAULT_MAX_POINT_CLOUD_POINTS,
        help="Maximum number of point-cloud points to plot per scene.",
    )
    return parser.parse_args(argv)


def scene_export_dir(root: str | Path, scene: str, split: str) -> Path:
    return Path(root) / scene / split / "exports"


def scene_has_required_exports(root: str | Path, scene: str, split: str) -> bool:
    export_dir = scene_export_dir(root, scene, split)
    required = (
        export_dir / "mini_npz" / "results.npz",
        export_dir / "gt_meta.npz",
        export_dir / "fuse" / "pcd.ply",
    )
    return all(path.exists() for path in required)


def discover_shared_scenes(
    *,
    da3_root: str | Path,
    pred_root: str | Path,
    split: str,
) -> list[str]:
    da3_root = Path(da3_root)
    pred_root = Path(pred_root)
    da3_scenes = {
        path.name
        for path in da3_root.iterdir()
        if path.is_dir() and scene_has_required_exports(da3_root, path.name, split)
    }
    pred_scenes = {
        path.name
        for path in pred_root.iterdir()
        if path.is_dir() and scene_has_required_exports(pred_root, path.name, split)
    }
    return sorted(da3_scenes & pred_scenes)


def filter_scenes_to_process(
    *,
    scenes: Iterable[str],
    pred_root: str | Path,
    split: str,
    skip_scenes: Iterable[str],
    overwrite: bool,
    static_name: str = DEFAULT_STATIC_XYTOP_NAME,
) -> list[str]:
    skip_set = set(skip_scenes)
    selected: list[str] = []
    for scene in scenes:
        if scene in skip_set:
            continue
        if not overwrite:
            static_path = scene_export_dir(pred_root, scene, split) / static_name
            if static_path.exists():
                continue
        selected.append(scene)
    return selected


def build_frustum(ext_w2c: np.ndarray, intrinsic: np.ndarray, depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2w = affine_inverse_np(ext_w2c[None])[0]
    fx = float(max(intrinsic[0, 0], 1e-6))
    fy = float(max(intrinsic[1, 1], 1e-6))
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    width = max(2.0 * cx, 1.0)
    height = max(2.0 * cy, 1.0)
    corners_cam = np.array(
        [
            [(-cx) / fx * depth, (-cy) / fy * depth, depth, 1.0],
            [((width - cx) / fx) * depth, (-cy) / fy * depth, depth, 1.0],
            [((width - cx) / fx) * depth, ((height - cy) / fy) * depth, depth, 1.0],
            [(-cx) / fx * depth, ((height - cy) / fy) * depth, depth, 1.0],
        ],
        dtype=np.float32,
    )
    origin = c2w[:3, 3]
    corners_world = (c2w @ corners_cam.T).T[:, :3]
    forward_world = (c2w @ np.array([0.0, 0.0, depth * 1.3, 1.0], dtype=np.float32))[:3]
    return origin, corners_world, forward_world


def draw_frustum_3d(ax, origin: np.ndarray, corners: np.ndarray, forward: np.ndarray, color: str, *, alpha: float, lw: float) -> None:
    ring = np.vstack([corners, corners[:1]])
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, alpha=alpha, lw=lw)
    for corner in corners:
        ax.plot(
            [origin[0], corner[0]],
            [origin[1], corner[1]],
            [origin[2], corner[2]],
            color=color,
            alpha=alpha,
            lw=lw,
        )
    ax.plot(
        [origin[0], forward[0]],
        [origin[1], forward[1]],
        [origin[2], forward[2]],
        color=color,
        alpha=min(1.0, alpha + 0.1),
        lw=lw + 0.3,
    )


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def set_axes_equal_2d(ax, xy_points: np.ndarray) -> None:
    mins = xy_points.min(axis=0)
    maxs = xy_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius <= 0:
        radius = 1.0
    margin = radius * 0.08
    ax.set_xlim(center[0] - radius - margin, center[0] + radius + margin)
    ax.set_ylim(center[1] - radius - margin, center[1] + radius + margin)
    ax.set_aspect("equal", adjustable="box")


def annotate_xy(ax, point_xy: np.ndarray, text: str, color: str, *, dx: float = 0.2, dy: float = 0.2, alpha: float = 0.95) -> None:
    ax.text(point_xy[0] + dx, point_xy[1] + dy, text, color=color, fontsize=9, alpha=alpha)


def load_scene_bundle(
    *,
    scene: str,
    da3_root: str | Path,
    pred_root: str | Path,
    split: str,
    max_point_cloud_points: int,
) -> dict[str, object]:
    da3_export = scene_export_dir(da3_root, scene, split)
    pred_export = scene_export_dir(pred_root, scene, split)

    da3_results = np.load(da3_export / "mini_npz" / "results.npz")
    pred_results = np.load(pred_export / "mini_npz" / "results.npz")
    gt_meta = np.load(pred_export / "gt_meta.npz", allow_pickle=True)

    da3_ext = as_homogeneous(np.asarray(da3_results["extrinsics"], dtype=np.float32))
    da3_intr = np.asarray(da3_results["intrinsics"], dtype=np.float32)
    pred_ext = as_homogeneous(np.asarray(pred_results["extrinsics"], dtype=np.float32))
    pred_intr = np.asarray(pred_results["intrinsics"], dtype=np.float32)
    gt_ext = as_homogeneous(np.asarray(gt_meta["extrinsics"], dtype=np.float32))
    gt_intr = np.asarray(gt_meta["intrinsics"], dtype=np.float32)

    _, _, da3_scale, da3_ext_aligned = align_poses_umeyama(
        gt_ext.copy(),
        da3_ext.copy(),
        return_aligned=True,
        ransac=len(gt_ext) >= 10,
        random_state=42,
    )
    _, _, pred_scale, pred_ext_aligned = align_poses_umeyama(
        gt_ext.copy(),
        pred_ext.copy(),
        return_aligned=True,
        ransac=len(gt_ext) >= 10,
        random_state=42,
    )

    trajectories = {
        "da3": {
            "ext": np.asarray(da3_ext_aligned, dtype=np.float32),
            "intr": da3_intr,
            "centers": affine_inverse_np(np.asarray(da3_ext_aligned, dtype=np.float32))[:, :3, 3],
            "scale": float(da3_scale),
        },
        "pred": {
            "ext": np.asarray(pred_ext_aligned, dtype=np.float32),
            "intr": pred_intr,
            "centers": affine_inverse_np(np.asarray(pred_ext_aligned, dtype=np.float32))[:, :3, 3],
            "scale": float(pred_scale),
        },
        "gt": {
            "ext": gt_ext,
            "intr": gt_intr,
            "centers": affine_inverse_np(gt_ext)[:, :3, 3],
            "scale": 1.0,
        },
    }

    num_frames = min(len(item["ext"]) for item in trajectories.values())
    for item in trajectories.values():
        item["ext"] = item["ext"][:num_frames]
        item["intr"] = item["intr"][:num_frames]
        item["centers"] = item["centers"][:num_frames]

    point_cloud = o3d.io.read_point_cloud(str(pred_export / "fuse" / "pcd.ply"))
    points = np.asarray(point_cloud.points)
    point_colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
    if len(points) > max_point_cloud_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(points), size=max_point_cloud_points, replace=False)
        points = points[keep]
        if point_colors is not None and len(point_colors) == len(point_cloud.points):
            point_colors = point_colors[keep]

    all_points = [points] if len(points) else []
    all_points.extend(item["centers"] for item in trajectories.values())
    all_points = np.concatenate(all_points, axis=0)
    scene_diag = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))

    return {
        "scene": scene,
        "num_frames": num_frames,
        "trajectories": trajectories,
        "points": points,
        "point_colors": point_colors,
        "all_points": all_points,
        "frustum_scale": max(scene_diag * 0.03, 0.08),
    }


def draw_scene_3d(ax, bundle: dict[str, object], *, upto: int, dynamic: bool, colors: dict[str, str], short_names: dict[str, str]) -> None:
    points = bundle["points"]
    point_colors = bundle["point_colors"]
    all_points = bundle["all_points"]
    trajectories = bundle["trajectories"]
    frustum_scale = float(bundle["frustum_scale"])

    if len(points):
        if point_colors is not None and len(point_colors) == len(points):
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=0.15, alpha=0.08)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#9aa0a6", s=0.15, alpha=0.08)

    for label in ("da3", "pred", "gt"):
        centers = trajectories[label]["centers"]
        color = colors[label]
        prefix = short_names[label]
        shown = centers[: upto + 1]
        ax.plot(shown[:, 0], shown[:, 1], shown[:, 2], color=color, lw=2.2 if label != "gt" else 2.0, alpha=0.95)
        ax.scatter(centers[0, 0], centers[0, 1], centers[0, 2], color=color, s=28, marker="o", alpha=0.95)
        ax.text(centers[0, 0], centers[0, 1], centers[0, 2], f" {prefix}-start", color=color, fontsize=9)
        ax.scatter(
            centers[-1, 0],
            centers[-1, 1],
            centers[-1, 2],
            facecolors="none" if dynamic else color,
            edgecolors=color,
            s=60,
            marker="^",
            alpha=0.95,
        )
        ax.text(centers[-1, 0], centers[-1, 1], centers[-1, 2], f" {prefix}-end", color=color, fontsize=9, alpha=0.85)
        if dynamic:
            current = centers[upto]
            ax.scatter(current[0], current[1], current[2], color=color, s=42, marker="s", alpha=1.0)
            ax.text(current[0], current[1], current[2], f" {prefix}-t{upto}", color=color, fontsize=8)
            origin, corners, forward = build_frustum(
                trajectories[label]["ext"][upto],
                trajectories[label]["intr"][upto],
                frustum_scale,
            )
            draw_frustum_3d(ax, origin, corners, forward, color, alpha=0.9, lw=1.2)
        else:
            for idx in range(0, int(bundle["num_frames"]), 4):
                origin, corners, forward = build_frustum(
                    trajectories[label]["ext"][idx],
                    trajectories[label]["intr"][idx],
                    frustum_scale,
                )
                draw_frustum_3d(ax, origin, corners, forward, color, alpha=0.35, lw=0.8)

    ax.set_title("Perspective view (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal_3d(ax, all_points)
    ax.view_init(elev=18, azim=-58)


def draw_scene_xy(ax, bundle: dict[str, object], *, upto: int, dynamic: bool, colors: dict[str, str], short_names: dict[str, str]) -> None:
    points = bundle["points"]
    point_colors = bundle["point_colors"]
    all_points = bundle["all_points"]
    trajectories = bundle["trajectories"]
    frustum_scale = float(bundle["frustum_scale"])

    if len(points):
        if point_colors is not None and len(point_colors) == len(points):
            ax.scatter(points[:, 0], points[:, 1], c=point_colors, s=0.15, alpha=0.08)
        else:
            ax.scatter(points[:, 0], points[:, 1], color="#9aa0a6", s=0.15, alpha=0.08)

    for label in ("da3", "pred", "gt"):
        centers = trajectories[label]["centers"]
        color = colors[label]
        prefix = short_names[label]
        shown = centers[: upto + 1]
        ax.plot(shown[:, 0], shown[:, 1], color=color, lw=2.2 if label != "gt" else 2.0, alpha=0.95)
        ax.scatter(centers[0, 0], centers[0, 1], color=color, s=28, marker="o", alpha=0.95)
        annotate_xy(ax, centers[0, :2], f"{prefix}-start", color)
        ax.scatter(
            centers[-1, 0],
            centers[-1, 1],
            facecolors="none" if dynamic else color,
            edgecolors=color,
            s=60,
            marker="^",
            alpha=0.95,
        )
        annotate_xy(ax, centers[-1, :2], f"{prefix}-end", color, dy=-0.4, alpha=0.85)
        if dynamic:
            current = centers[upto]
            ax.scatter(current[0], current[1], color=color, s=42, marker="s", alpha=1.0)
            annotate_xy(ax, current[:2], f"{prefix}-t{upto}", color, dx=0.25, dy=0.25)
            origin, _, forward = build_frustum(
                trajectories[label]["ext"][upto],
                trajectories[label]["intr"][upto],
                frustum_scale,
            )
            direction = forward[:2] - origin[:2]
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction = direction / norm * max(0.7, frustum_scale * 0.6)
                ax.arrow(
                    origin[0],
                    origin[1],
                    direction[0],
                    direction[1],
                    color=color,
                    width=0.03,
                    head_width=0.35,
                    length_includes_head=True,
                    alpha=0.9,
                )
        else:
            for idx in range(0, int(bundle["num_frames"]), 6):
                origin, _, forward = build_frustum(
                    trajectories[label]["ext"][idx],
                    trajectories[label]["intr"][idx],
                    frustum_scale,
                )
                direction = forward[:2] - origin[:2]
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    direction = direction / norm * max(0.6, frustum_scale * 0.45)
                    ax.arrow(
                        origin[0],
                        origin[1],
                        direction[0],
                        direction[1],
                        color=color,
                        width=0.02,
                        head_width=0.25,
                        length_includes_head=True,
                        alpha=0.28,
                    )

    ax.set_title("Top view (XY orthographic)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    set_axes_equal_2d(ax, all_points[:, :2])
    ax.grid(True, alpha=0.3)


def make_figure(bundle: dict[str, object], *, upto: int, dynamic: bool) -> plt.Figure:
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    colors = {"da3": "#ff7f0e", "pred": "#1f77b4", "gt": "#2ca02c"}
    short_names = {"da3": "D", "pred": "P", "gt": "G"}
    draw_scene_3d(ax1, bundle, upto=upto, dynamic=dynamic, colors=colors, short_names=short_names)
    draw_scene_xy(ax2, bundle, upto=upto, dynamic=dynamic, colors=colors, short_names=short_names)

    trajectories = bundle["trajectories"]
    handles = [
        Line2D([0], [0], color=colors["da3"], lw=2, label="DA3 anyview (aligned, s={:.2f})".format(trajectories["da3"]["scale"])),
        Line2D([0], [0], color=colors["pred"], lw=2, label="Stitched pred (aligned, s={:.2f})".format(trajectories["pred"]["scale"])),
        Line2D([0], [0], color=colors["gt"], lw=2, label="GT"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    suffix = "Frame {}/{}".format(upto + 1, bundle["num_frames"]) if dynamic else "Full trajectory"
    fig.suptitle("ETH3D {} camera trajectories ({})".format(bundle["scene"], suffix), fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


def render_scene_visualization(
    *,
    scene: str,
    da3_root: str | Path,
    pred_root: str | Path,
    split: str,
    fps: int,
    max_point_cloud_points: int,
) -> dict[str, Path]:
    pred_export_dir = scene_export_dir(pred_root, scene, split)
    static_path = pred_export_dir / DEFAULT_STATIC_XYTOP_NAME
    video_path = pred_export_dir / DEFAULT_VIDEO_XYTOP_NAME
    frames_dir = pred_export_dir / DEFAULT_FRAMES_XYTOP_DIRNAME
    meta_path = pred_export_dir / DEFAULT_META_XYTOP_NAME
    frames_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_scene_bundle(
        scene=scene,
        da3_root=da3_root,
        pred_root=pred_root,
        split=split,
        max_point_cloud_points=max_point_cloud_points,
    )

    static_fig = make_figure(bundle, upto=int(bundle["num_frames"]) - 1, dynamic=False)
    static_fig.savefig(static_path, dpi=220)
    plt.close(static_fig)

    video_size = (3200, 1600)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, video_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    for frame_idx in range(int(bundle["num_frames"])):
        fig = make_figure(bundle, upto=frame_idx, dynamic=True)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        rgb = rgba[:, :, :3].copy()
        if width != video_size[0] or height != video_size[1]:
            rgb = cv2.resize(rgb, video_size, interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame_path = frames_dir / f"frame_{frame_idx:03d}.png"
        cv2.imwrite(str(frame_path), bgr)
        writer.write(bgr)
        plt.close(fig)

    writer.release()

    trajectories = bundle["trajectories"]
    np.savez_compressed(
        meta_path,
        split=np.array(split),
        num_frames=np.array(bundle["num_frames"], dtype=np.int32),
        da3_extrinsics_aligned=trajectories["da3"]["ext"].astype(np.float32),
        pred_extrinsics_aligned=trajectories["pred"]["ext"].astype(np.float32),
        gt_extrinsics=trajectories["gt"]["ext"].astype(np.float32),
        da3_centers=trajectories["da3"]["centers"].astype(np.float32),
        pred_centers=trajectories["pred"]["centers"].astype(np.float32),
        gt_centers=trajectories["gt"]["centers"].astype(np.float32),
        xy_min=bundle["all_points"][:, :2].min(axis=0).astype(np.float32),
        xy_max=bundle["all_points"][:, :2].max(axis=0).astype(np.float32),
        da3_umeyama_scale=np.asarray(trajectories["da3"]["scale"], dtype=np.float32),
        pred_umeyama_scale=np.asarray(trajectories["pred"]["scale"], dtype=np.float32),
    )

    return {
        "static": static_path,
        "video": video_path,
        "frames_dir": frames_dir,
        "meta": meta_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    da3_root = Path(args.da3_root)
    pred_root = Path(args.pred_root)
    if not da3_root.exists():
        raise FileNotFoundError(f"DA3 root does not exist: {da3_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {pred_root}")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_point_cloud_points <= 0:
        raise ValueError("--max_point_cloud_points must be positive")

    shared_scenes = discover_shared_scenes(
        da3_root=da3_root,
        pred_root=pred_root,
        split=args.split,
    )
    if args.scenes:
        shared_set = set(shared_scenes)
        shared_scenes = [scene for scene in args.scenes if scene in shared_set]

    target_scenes = filter_scenes_to_process(
        scenes=shared_scenes,
        pred_root=pred_root,
        split=args.split,
        skip_scenes=args.skip_scenes,
        overwrite=args.overwrite,
    )

    print(f"shared scenes: {len(shared_scenes)}")
    print(f"target scenes: {len(target_scenes)}")
    if not target_scenes:
        print("nothing to render")
        return 0

    for index, scene in enumerate(target_scenes, start=1):
        print(f"[{index}/{len(target_scenes)}] rendering {scene}")
        outputs = render_scene_visualization(
            scene=scene,
            da3_root=da3_root,
            pred_root=pred_root,
            split=args.split,
            fps=args.fps,
            max_point_cloud_points=args.max_point_cloud_points,
        )
        print(f"  static: {outputs['static']}")
        print(f"  video:  {outputs['video']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
