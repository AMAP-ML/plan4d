# Depth Anything 3 For Plan4D

This directory contains the DA3 backbone, depth/ray/camera decoders, token-only
scene-flow head, Python API, video inference tools, training, and evaluation.

- [Scene-flow design](SCENE_FLOW.md)
- [Training and data contracts](training/README.md)
- [Plan4D integration](../README.md)

From the Plan4D root, use `bash scripts/train_scene_flow.sh` for training and
`bash scripts/infer_scene_flow.sh --video_path /path/to/video.mp4` for DA3
scene-flow inference. Use `--image_dir` instead for an RGB frame directory.
The inference script loads the trained head from `checkpoints/scene_flow_head.bin`
by default; override it with `--scene_flow_head_checkpoint_path`.

Install from the Plan4D root with `python -m pip install -e ./Depth-Anything-3`.
Supply pretrained weights separately. The original DA3 license is preserved
in `LICENSE`; upstream DA3 is maintained at
https://github.com/ByteDance-Seed/Depth-Anything-3.
