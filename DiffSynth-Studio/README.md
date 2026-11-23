# DiffSynth Runtime For Plan4D

This directory includes the DiffSynth runtime used by Plan4D's LongCat/Wan
video generation and joint geometry training.

Primary entry points:

- `examples/wanvideo/model_training/train.py`
- `examples/wanvideo/model_inference/LongCat-Video-stitched-joint.py`
- `examples/wanvideo/model_inference/LongCat-Video-stitched-joint-sequential-batch-manifest.py`

LongCat's DiT is implemented in `diffsynth/models/longcat_video_dit.py`.
Wan models, the VAE, and the shared LongCat/Wan pipeline are under
`diffsynth/models/` and `diffsynth/pipelines/wan_video.py`.

See the [Plan4D README](../README.md) for setup, weights, and workflows.
Training settings are centralized in `../configs/joint.yaml`; launch with
`bash scripts/train_joint.sh` from the Plan4D root. Use
`bash scripts/infer_joint.sh` for the trained joint model or
`bash scripts/infer_pipeline.sh` for the complete RGB-and-geometry pipeline.
Both inference scripts accept a leading `--batch` for JSONL input, or native
single-sample arguments such as `--prompt` and `--condition_path`.
The manifest runner
replaces dataset-specific batch scripts. Only the LongCat/Wan runtime and its
transitive dependencies are included.
The existing upstream Apache-2.0 license is preserved in `LICENSE`.
