# DA3 Scene-Flow Head Training

This training package freezes the pretrained DA3 model and trains the token-only
`scene_flow_head`. It uses the shared interface in `plan4d/da3_interface.py`.

## Multi-dataset Training

The sole training configuration is `../../configs/scene_flow.yaml`.
A single mixed DataLoader and training loop combine the supported datasets.

- PointOdyssey, BlinkVision, and Virtual KITTI2 provide direct motion supervision.
- OmniWorld reconstructs supervision from depth, pose, and optical flow.
- Dynamic Replica and MOVi-F have dedicated adapters and evaluation support.
- The default target is relative scene flow from frame `t` to frame `t+1`, expressed in the first camera's coordinate basis.
- Dataset outputs use `[T, H, W, 3]` flow and `[T, H, W]` `deform_mask`.
- DA3 remains frozen; only the scene-flow head is trainable.

Different datasets provide different supervision sources, but their adapters
produce the same fields. PointOdyssey has no native dense scene-flow labels:
its sparse 3D trajectories are rasterized into a dense-shaped tensor with a
sparse validity mask. The loss only uses pixels where `deform_mask` is valid.

## Training Pipeline

1. Load DA3 from `config.json` and `model.safetensors` in the configured weights directory.
2. Inject `scene_flow_head` and `scene_flow_adapter` into the anyview branch using the `da3-giant-sceneflow` preset.
3. Load pretrained weights with `strict=False` to allow the new head.
4. Freeze all DA3 parameters, then unfreeze `scene_flow_head`.
5. Extract four encoder feature levels, decode with `output_basis="frame0"`, and compute masked scene-flow loss.

The decoder consumes four encoder token levels and has no RGB fusion branch.

## Data Contract

| Field | Shape | Meaning |
| --- | --- | --- |
| `rgbs` | `[T, H, W, 3]` | RGB images, uint8 |
| `intrinsics` | `[T, 3, 3]` | Camera intrinsics |
| `extrinsics` | `[T, 4, 4]` | World-to-camera transforms |
| `scene_flow` | `[T, H, W, 3]` | Flow vectors in the first camera basis |
| `scene_flow_scale` | scalar | Per-chunk flow normalization scale |
| `deform_mask` | `[T, H, W]` | Valid supervision pixels |
| `sequence_length` | scalar | Number of frames |

For PointOdyssey, `anno.npz` contains world-coordinate `trajs_3d`. Compute
`delta_world[t] = X_world[t+1] - X_world[t]`, rotate by the first frame's
world-to-camera rotation, and rasterize the vectors at `trajs_2d[t]`.

With `flow_target_space=relative`, normalize metric flow using the configured
per-chunk scale. The published configuration uses `geometry_radius` with
`mean_radius`; flow-magnitude statistics such as `p95` remain available through
`relative_scale_source=flow_p95`.

A valid track must be valid in the current and next frames, visible in the
current frame when visibility labels exist, and inside the resized/cropped
image. The final frame has both zero flow and zero `deform_mask`.

## Loss

`training/losses/scene_flow_loss.py` implements masked L1 or EPE regression,
optional final-frame zero regularization, and optional depth-warp supervision.
Regression uses the supplied validity mask and does not silently supervise the
entire image. Targets are relative by default; `flow_target_space=metric`
selects metric targets. Depth-warp supervision requires consistent flow,
depth, and camera-translation scales.

## Commands

Run these commands from the Plan4D root after activating your
environment. Configure dataset and model paths before launching training.

```bash
python Depth-Anything-3/training/train_scene_flow_head.py --help
NUM_PROCESSES=4 bash scripts/train_scene_flow.sh
bash scripts/infer_scene_flow.sh --video_path /path/to/video.mp4
```

Use the same configuration for evaluation:

```bash
python Depth-Anything-3/training/eval_scene_flow_head.py \
  --config ../configs/scene_flow.yaml --checkpoint /path/to/head_checkpoint
```

Enable only the required entries in `data.datasets` for a single-dataset run.
Paths inside this native DA3 configuration are relative to `Depth-Anything-3`.
Resume with `bash scripts/train_scene_flow.sh -- --resume /path/to/checkpoint`.

## Two Chunk Sizes

`data.chunk_size` controls the sequence length of each dataset sample.
`model.decoder_chunk_size` controls temporal microbatching inside the DPT
decoder to reduce peak memory. They need not be equal. A smaller decoder
chunk size splits a sample into temporal decoding batches; a size at least as
large as the sample decodes the sequence together.

## Checkpoints And Extensions

The package includes mixture sampling, distributed training, relative target
normalization, checkpoint loading, and evaluation. Checkpoints store the head
and training state; base DA3 weights remain a separate prerequisite.

Potential extensions include camera-basis auxiliary supervision, backward
flow, confidence prediction, and denser trajectory splatting. These are not
part of the current token-only three-channel head.
