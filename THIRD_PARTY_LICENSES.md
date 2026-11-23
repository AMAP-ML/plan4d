# Third-Party Licenses and Attributions

This project incorporates and builds upon code from the following third-party
projects. We thank their authors for releasing their work. Each entry lists
the license that governs the corresponding code in this repository.

## Bundled components (subdirectories with their own LICENSE files)

### Depth-Anything-3 — Apache License 2.0
- Source: https://github.com/ByteDance-Seed/Depth-Anything-3
- Location: `Depth-Anything-3/`
- The upstream `LICENSE` file is retained in that directory. Files carry the
  original `Copyright (c) 2025 ByteDance Ltd. and/or its affiliates` headers.
- Code inherited from upstream DA3's own dependencies retains its upstream
  attribution inside the file headers and comments:
  - `model/dinov2/` — DINO ViT backbone implementation referencing
    [facebookresearch/dino](https://github.com/facebookresearch/dino)
    (Apache-2.0) and
    [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) /
    [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
    (timm, Apache-2.0); RoPE layer comments reference
    [meta-llama/codellama](https://github.com/meta-llama/codellama) and
    [naver-ai/rope-vit](https://github.com/naver-ai/rope-vit) (upstream DA3
    attribute comments are kept as-is; Meta and Naver retain their respective
    rights in the referenced material).
  - `utils/read_write_model.py` — COLMAP model I/O, `Copyright (c), ETH Zurich
    and UNC Chapel Hill` (BSD-3-Clause, notice retained in the file header).
  - `model/utils/head_utils.py` — states "Inspired by
    [microsoft/moge](https://github.com/microsoft/moge)" (MIT).
  - `utils/sh_helpers.py` — references a spherical-harmonics conversion
    formula from a [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
    issue; formula reference only, no code copied.

### DiffSynth-Studio — Apache License 2.0
- Source: https://github.com/modelscope/DiffSynth-Studio
- Location: `DiffSynth-Studio/`
- The upstream `LICENSE` file is retained in that directory.
- Files inherited from upstream DiffSynth-Studio's own dependencies:
  - `models/wan_video_image_encoder.py` — CLIP image encoder based on
    [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
    (Apache-2.0) / [openai/CLIP](https://github.com/openai/CLIP) (MIT).
  - `models/wan_video_camera_controller.py` — camera-pose utilities
    "Copied from" [hehao13/CameraCtrl](https://github.com/hehao13/CameraCtrl)
    (Apache-2.0), as noted in the file docstrings.

## Per-file attributions (repo root)

### `utils/lora_util/` — MIT License

LoRA layer implementation from
[microsoft/LoRA](https://github.com/microsoft/LoRA), with the original
copyright headers retained in `utils/lora_util/layers.py` and
`utils/lora_util/utils.py`. The MIT license text is reproduced below as
required by its notice-retention clause:

```
Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### `utils/wan_utils.py` — Apache License 2.0 (diffusers)

The Wan 2.1 VAE implementation (WanCausalConv3d through AutoencoderKLWan) is
modified from `diffusers/models/autoencoders/autoencoder_kl_wan.py` in
[huggingface/diffusers](https://github.com/huggingface/diffusers)
(Apache-2.0), as noted in the file header.

### `data/image_preprocessing.py`

Image loading and resize/crop helpers written for this project; they preserve
the numerical behavior (isotropic short-side resize, bilinear interpolation,
centre crop) used by the training and evaluation pipelines.

## Pretrained models and datasets

Pretrained checkpoints (LongCat-Video, Wan 2.1 VAE, Depth Anything 3,
CLIP-based scorers) and datasets (PointOdyssey, BlinkVision, Virtual KITTI2,
OmniWorld, Dynamic Replica, MOVi-F, MVS-Synth, Action100M, LingBot-Depth,
DTU, ETH3D, ScanNet++, SevenScenes, HiRoom) remain subject to their own
licenses and access conditions.

## Generation-quality evaluation

The generation-quality evaluation scripts (CLIP score, LongCLIP score,
aesthetic quality, MUSIQ imaging quality, UnifiedReward, DPG-Bench) are not
distributed with this release. To reproduce those metrics, use the official
implementations under their respective licenses:

- CLIP score: `torchmetrics` `CLIPScore` (Apache-2.0) or `openai/CLIP` (MIT)
- MUSIQ / aesthetic quality: [chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
  (Apache-2.0) / [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)
- UnifiedReward: [CodeGoat24/UnifiedReward](https://github.com/CodeGoat24/UnifiedReward)
- DPG-Bench: the official DPG evaluation protocol and model answers
