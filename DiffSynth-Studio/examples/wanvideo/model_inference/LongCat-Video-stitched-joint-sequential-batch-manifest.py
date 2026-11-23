"""
Single-node multi-GPU batch inference for manifest JSONL datasets.
Load models once per GPU, then iterate over the assigned task subset.

Usage:
    python LongCat-Video-stitched-joint-sequential-batch-manifest.py \
        --batch_input_jsonl /path/to/manifest.jsonl \
        --manifest_data_root /path/to/data_root \
        --gpus 0,1,2,3 \
        --output_dir /path/to/output \
        ...

Manifest JSONL format (one JSON object per line): each entry must have at least:
    - sample_id      (str)
    - prompt         (str)
    - frame_indices  (list[int])

Current implementation only supports ti2v entries with cond_frames=1.
The condition image can be provided directly through condition_path. Otherwise,
the first-frame path is derived as:
    {manifest_data_root}/{rgb_relpath}/{frame_indices[0]:06d}.png
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.multiprocessing as mp

THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
DIFFSYNTH_ROOT = THIS_FILE.parents[3]
PROJECT_ROOT = THIS_FILE.parents[4]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_ROOT = PLAN4D_ROOT / "Depth-Anything-3"
DA3_SRC_ROOT = DA3_ROOT / "src"

for candidate in (DIFFSYNTH_ROOT, PROJECT_ROOT, DA3_ROOT, DA3_SRC_ROOT, PLAN4D_ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import importlib

_orig_spec = importlib.util.spec_from_file_location(
    "longcat_stitched_joint",
    SCRIPT_DIR / "LongCat-Video-stitched-joint.py",
)
_orig_module = importlib.util.module_from_spec(_orig_spec)
sys.modules["longcat_stitched_joint"] = _orig_module
_orig_spec.loader.exec_module(_orig_module)

from longcat_stitched_joint import (
    DEFAULT_DA3_MODEL_PATH,
    DEFAULT_DIT_CHECKPOINT_PATH,
    DEFAULT_DIT_LORA_ALPHA,
    DEFAULT_DIT_LORA_RANK,
    DEFAULT_DIT_LORA_TARGET_MODULES,
    DEFAULT_LONGCAT_VAE_FILE,
    DEFAULT_LONGCAT_WEIGHT_ROOT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STITCHED_DECODER_CHECKPOINT_PATH,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_WAN_VAE_PATH,
    build_generation_shape_notice,
    build_stitched_decoder_checkpoint_plan,
    generate_video_and_latents,
    load_condition_frames,
    load_longcat_pipeline,
    load_stitched_decoder,
    resolve_dit_checkpoint_override,
    run_stitched_geometry,
    save_outputs,
)
from models.stitching_layer_builder import parse_conv_spec
from depth_anything_3.utils.scene_flow_postprocess import scene_flow_filter_period


def sanitize_run_name(sample_id: str) -> str:
    return sample_id.replace("/", "_").replace("\\", "_")


def load_batch_tasks(jsonl_path: str, manifest_data_root: str) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    root_path = Path(manifest_data_root)

    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            task_type = entry.get("task_type", "ti2v")
            cond_frames = int(entry.get("cond_frames", 1))
            if task_type != "ti2v" or cond_frames != 1:
                raise ValueError(
                    "Only ti2v manifest entries with cond_frames=1 are supported, "
                    f"got task_type={task_type!r}, cond_frames={cond_frames}"
                )

            frame_indices = entry["frame_indices"]
            if not frame_indices:
                raise ValueError(f"frame_indices must be non-empty for sample_id={entry.get('sample_id')}")
            raw_condition_path = entry.get("condition_path") or entry.get("first_frame_path")
            if raw_condition_path:
                condition_path = Path(str(raw_condition_path)).expanduser()
                if not condition_path.is_absolute():
                    condition_path = root_path / condition_path
            else:
                first_frame_index = int(frame_indices[0])
                condition_path = root_path / entry["rgb_relpath"] / f"{first_frame_index:06d}.png"

            manifest_meta = {
                "sample_id": entry.get("sample_id"),
                "dataset": entry.get("dataset") or entry.get("dataset_name"),
                "scene_id": entry.get("scene_id"),
                "scene_relpath": entry.get("scene_relpath"),
                "rgb_relpath": entry.get("rgb_relpath"),
                "text_relpath": entry.get("text_relpath"),
                "frame_indices": frame_indices,
                "source_fps": entry.get("source_fps"),
                "task_type": task_type,
                "cond_frames": cond_frames,
                "episode_index": entry.get("episode_index"),
                "gt_video_path": entry.get("gt_video_path"),
                "reference_video_path": entry.get("reference_video_path"),
                "raw_frame_count": entry.get("raw_frame_count"),
                "prompt_is_fallback": entry.get("prompt_is_fallback"),
                "is_padded": entry.get("is_padded"),
                "camera_key": entry.get("camera_key"),
                "fps": entry.get("fps"),
            }

            tasks.append(
                {
                    "index": line_number,
                    "sample_id": entry["sample_id"],
                    "prompt": entry["prompt"],
                    "condition_path": str(condition_path),
                    "run_name": str(entry.get("run_name") or sanitize_run_name(entry["sample_id"])),
                    "manifest_meta": manifest_meta,
                }
            )

    return tasks


def split_tasks_across_gpus(
    tasks: list[dict[str, Any]],
    num_gpus: int,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(num_gpus)]
    for task_index, task in enumerate(tasks):
        chunks[task_index % num_gpus].append(task)
    return chunks


def save_outputs_with_manifest_metadata(
    args: argparse.Namespace,
    output_dir: Path,
    generation: dict[str, Any],
    geometry: dict[str, Any],
    condition_frames,
    task: dict[str, Any],
    batch_input_jsonl: str,
) -> None:
    save_outputs(args, output_dir, generation, geometry, condition_frames)

    meta_path = output_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)

    manifest_meta = dict(task.get("manifest_meta", {}))
    sample_id = task.get("sample_id")
    if sample_id is not None:
        manifest_meta["sample_id"] = sample_id
    manifest_meta["manifest_path"] = batch_input_jsonl

    meta.update({key: value for key, value in manifest_meta.items() if value is not None})

    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-node multi-GPU batch inference for manifest JSONL datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch_input_jsonl", type=str, required=True)
    parser.add_argument("--manifest_data_root", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU indices.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--task_type", type=str, default="ti2v", choices=["ti2v", "continuation", "t2v"])
    parser.add_argument("--continuation_cond_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=504)
    parser.add_argument("--width", type=int, default=504)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--sigma_shift", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--switch_dit_boundary", type=float, default=0.875)
    parser.add_argument("--rand_device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument("--save_latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_point_map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_point_cloud_ply", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_merged_point_cloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_points_per_frame", type=int, default=200000)
    parser.add_argument("--max_points_merged", type=int, default=1000000)
    parser.add_argument("--longcat_weight_root", type=str, default=DEFAULT_LONGCAT_WEIGHT_ROOT)
    parser.add_argument("--longcat_vae_file", type=str, default=DEFAULT_LONGCAT_VAE_FILE)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--checkpoint_path",
        "--dit_checkpoint_path",
        dest="dit_checkpoint_path",
        type=str,
        default=DEFAULT_DIT_CHECKPOINT_PATH,
    )
    parser.add_argument("--dit_lora_path", type=str, default=None)
    parser.add_argument("--dit_lora_base_checkpoint_path", type=str, default=None)
    parser.add_argument("--dit_lora_target_modules", type=str, default=DEFAULT_DIT_LORA_TARGET_MODULES)
    parser.add_argument("--dit_lora_rank", type=int, default=DEFAULT_DIT_LORA_RANK)
    parser.add_argument("--dit_lora_alpha", type=int, default=DEFAULT_DIT_LORA_ALPHA)
    parser.add_argument("--stitched_decoder_checkpoint_path", type=str, default=DEFAULT_STITCHED_DECODER_CHECKPOINT_PATH)
    parser.add_argument("--feedforward_model", type=str, default="da3", choices=["da3"])
    parser.add_argument("--video_model", type=str, default="wan", choices=["wan"])
    parser.add_argument("--da3_model_path", type=str, default=DEFAULT_DA3_MODEL_PATH)
    parser.add_argument("--da3_config_name", type=str, default="config.json")
    parser.add_argument("--da3_checkpoint_name", type=str, default="model.safetensors")
    parser.add_argument("--da3_ref_view_strategy", type=str, default="saddle_balanced")
    parser.add_argument("--da3_enable_scene_flow_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--da3_scene_flow_head_checkpoint_path", type=str, default=None)
    parser.add_argument("--da3_scene_flow_preset", type=str, default="da3-giant-sceneflow")
    parser.add_argument("--wan_vae_path", type=str, default=DEFAULT_WAN_VAE_PATH)
    parser.add_argument("--hf_endpoint", type=str, default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--wan_vae_repo_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--stitching_layer_location", type=str, default="enc_blocks_2")
    parser.add_argument(
        "--stitching_layer_config",
        type=parse_conv_spec,
        default=parse_conv_spec("conv3d_k5x3x3_o1536_s1x2x2_p2x1x1"),
        metavar="CONV_SPEC",
    )
    parser.add_argument("--lora_config", type=str, default="r64,a32,d0.0,f0")
    parser.add_argument("--full_finetune_stitched_parts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--feedforward_resolution", type=int, default=504)
    parser.add_argument("--da3_process_res_method", type=str, default="upper_bound_resize", choices=["upper_bound_resize"])
    parser.add_argument(
        "--scene_flow_degrid_period", "--scene-flow-degrid-period",
        type=scene_flow_filter_period, default=0.0, help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def gpu_worker(
    rank: int,
    gpu_ids: list[int],
    task_chunks: list[list[dict[str, Any]]],
    args: argparse.Namespace,
    total_tasks: int,
) -> None:
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)

    my_tasks = task_chunks[rank]
    if len(my_tasks) == 0:
        print(f"[GPU {gpu_id}] No tasks assigned, exiting.")
        return

    print(f"[GPU {gpu_id}] Assigned {len(my_tasks)} tasks.")

    worker_args = argparse.Namespace(**vars(args))
    worker_args.device = str(device)

    checkpoint_plan = build_stitched_decoder_checkpoint_plan(worker_args)

    print(f"[GPU {gpu_id}] Loading LongCat pipeline ...")
    if worker_args.dit_lora_path:
        base_dit = resolve_dit_checkpoint_override(worker_args) or str(Path(worker_args.longcat_weight_root) / "dit")
        print(f"[GPU {gpu_id}] Using base DiT for LoRA inference from {base_dit}")
        print(f"[GPU {gpu_id}] Applying DiT LoRA from {worker_args.dit_lora_path}")
    pipe = load_longcat_pipeline(worker_args)

    print(f"[GPU {gpu_id}] Loading stitched decoder ...")
    decoder_args = argparse.Namespace(**vars(worker_args))
    decoder_args.checkpoint_path = worker_args.stitched_decoder_checkpoint_path
    stitched_decoder = load_stitched_decoder(
        decoder_args,
        checkpoint_plan=checkpoint_plan,
        preserve_explicit_fields=(
            "feedforward_resolution",
            "resolution",
            "da3_enable_scene_flow_head",
            "da3_scene_flow_head_checkpoint_path",
            "da3_scene_flow_preset",
        ),
    )

    print(f"[GPU {gpu_id}] Models loaded. Starting inference on {len(my_tasks)} tasks.\n")

    completed = 0
    skipped = 0
    failed = 0

    for local_idx, task in enumerate(my_tasks):
        run_name = task["run_name"]
        prompt = task["prompt"]
        condition_path = task["condition_path"]
        output_dir = Path(worker_args.output_dir) / run_name

        if (output_dir / "meta.json").exists():
            skipped += 1
            print(f"[GPU {gpu_id}] [{local_idx + 1}/{len(my_tasks)}] SKIP '{run_name}' (already completed)")
            continue

        print(
            f"[GPU {gpu_id}] [{local_idx + 1}/{len(my_tasks)}] Processing '{run_name}'\n"
            f"  prompt: {prompt}\n"
            f"  condition: {condition_path}"
        )

        if not os.path.exists(condition_path):
            failed += 1
            print(f"  [ERROR] Condition path does not exist: {condition_path}. Skipping.\n")
            continue

        try:
            task_args = argparse.Namespace(**vars(worker_args))
            task_args.prompt = prompt
            task_args.condition_path = condition_path
            task_args.run_name = run_name

            condition_frames = load_condition_frames(task_args)
            if condition_frames is not None:
                print(f"  Loaded {len(condition_frames)} condition frame(s)")

            shape_notice = build_generation_shape_notice(task_args)
            if shape_notice is not None:
                print(f"  {shape_notice}")

            generation = generate_video_and_latents(pipe, task_args, condition_frames)
            geometry = run_stitched_geometry(task_args, stitched_decoder, generation["latents"])

            output_dir.mkdir(parents=True, exist_ok=True)
            save_outputs_with_manifest_metadata(
                task_args,
                output_dir,
                generation,
                geometry,
                condition_frames,
                task=task,
                batch_input_jsonl=worker_args.batch_input_jsonl,
            )

            completed += 1
            print(f"  [DONE] Saved to {output_dir}\n")

        except Exception as error:
            failed += 1
            print(f"  [ERROR] {error}")
            traceback.print_exc()
            print()
            continue

    print(
        f"[GPU {gpu_id}] Finished.\n"
        f"  Assigned: {len(my_tasks)}, Completed: {completed}, Skipped: {skipped}, Failed: {failed}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(",") if gpu_id.strip()]
    num_gpus = len(gpu_ids)
    if num_gpus == 0:
        raise ValueError("No GPU specified. Use --gpus to specify at least one GPU index.")

    tasks = load_batch_tasks(args.batch_input_jsonl, args.manifest_data_root)
    total_tasks = len(tasks)
    print(f"[INFO] Loaded {total_tasks} tasks from {args.batch_input_jsonl}")
    print(f"[INFO] Using {num_gpus} GPU(s): {gpu_ids}")
    print(f"[INFO] Manifest data root: {args.manifest_data_root}")
    print(f"[INFO] Output directory: {args.output_dir}")

    task_chunks = split_tasks_across_gpus(tasks, num_gpus)
    for chunk_index, chunk in enumerate(task_chunks):
        print(f"  GPU {gpu_ids[chunk_index]}: {len(chunk)} tasks")

    os.makedirs(args.output_dir, exist_ok=True)

    if num_gpus == 1:
        args.device = f"cuda:{gpu_ids[0]}"
        gpu_worker(0, gpu_ids, task_chunks, args, total_tasks)
    else:
        mp.spawn(
            gpu_worker,
            args=(gpu_ids, task_chunks, args, total_tasks),
            nprocs=num_gpus,
            join=True,
        )

    print(f"[INFO] All {total_tasks} tasks finished. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
