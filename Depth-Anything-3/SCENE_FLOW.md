# DA3 Scene-Flow Design And Usage

## Purpose

The `da3-giant-sceneflow` preset adds a token-only scene-flow decoder while
preserving DA3's existing depth, ray, pose, and Gaussian-splatting interfaces.
The head consumes four DINO encoder token levels and reuses DPT's projection,
resizing, and top-down fusion design. It predicts three flow channels without
an RGB merge branch, confidence head, backward flow, or RGB refinement.

## Geometric Meaning

`scene_flow[t]` is attached to the pixel grid of frame `t` and represents the
3D displacement from frame `t` to `t+1`:

```text
scene_flow[t] = X[t+1] - X[t]
```

All final flow vectors use the first camera's coordinate basis. The pixel
grid remains frame-local, while vector components share a common basis.
The last frame is explicitly zero because it has no next frame.

`scene_flow_cam` is the raw output in each frame's camera basis.
`scene_flow` is the output after rotation into the first-frame basis.
Extrinsics use the world-to-camera (`w2c`) convention. When extrinsics are
normalized to the first camera, the inverse rotation maps camera-basis
vectors to the first-frame basis. Vector transforms use rotation and scale,
never translation.

The basis convention does not determine physical scale. Metric targets teach
metric displacement; normalized targets teach relative displacement. The
published training configuration uses relative flow with a per-chunk
`geometry_radius` scale (`mean_radius`). Flow-magnitude normalization using
`flow_p95` is also supported. Absolute metric scale cannot
be recovered from those predictions without additional scale information.

## Implementation

- `src/depth_anything_3/model/sceneflowdpt.py`: `SceneFlowDPT` inherits DPT, fuses four token levels, and emits `scene_flow_cam` with three channels.
- `src/depth_anything_3/model/scene_flow_adapter.py`: rotates raw vectors to the first camera basis.
- `src/depth_anything_3/model/da3.py`: connects backbone, depth/ray heads, pose, scene flow, optional Gaussian splats, and sky postprocessing.
- `src/depth_anything_3/api.py`, `specs.py`, and `utils/io/output_processor.py`: expose `infer_scene_flow` and `Prediction.scene_flow`.
- `src/depth_anything_3/configs/da3-giant-sceneflow.yaml`: model preset.
- `../plan4d/da3_interface.py`: feature extraction and token-only decoding interface used by training.

When scene-flow inference is enabled without explicit extrinsics, DA3 uses
`ref_view_strategy="first"` to preserve the reference-frame convention.
`output.extrinsics` remains world-to-camera. When aligning predictions to
input extrinsic scale, the API applies the corresponding Sim(3) rotation and
scale to flow, without adding translation.

## Python API

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3(model_name="da3-giant-sceneflow").to("cuda")
prediction = model.inference(image=image_paths, infer_scene_flow=True)
scene_flow = prediction.scene_flow
```

This constructs the architecture; load the required pretrained DA3 and
trained scene-flow weights for meaningful inference. The returned NumPy
array has shape `(S, H, W, 3)` and a zero final frame.

For cached features:

```python
scene_flow = da3_interface.decode_features_to_scene_flow(
    features=features,
    image_size=(height, width),
    extrinsics=extrinsics_w2c,
    output_basis="frame0",
)
```

Features have shape `[B, 12288, S, ph, pw]`, consisting of four concatenated
3072-channel feature levels. The interface splits them, reconstructs token
tuples with dummy camera tokens, and calls the flow head. `output_basis="camera"`
returns raw camera-basis flow. `output_basis="frame0"` requires explicit
extrinsics and raises an error when they are absent.

## Compatibility And Training

Scene-flow inference is opt-in. Existing positional API parameters retain
their order, with `infer_scene_flow` appended. Existing keyword construction
of `Prediction` remains valid. Feature extraction and geometry decoding keep
their original contracts. `Prediction` exposes final flow, while internal
model outputs can also expose raw camera-basis flow.

Training and dataset integration are implemented in `training/`; see
`training/README.md`. Dataset adapters provide flow, validity masks, camera
parameters, and normalization metadata. The head is trained with the base
DA3 model frozen. Joint video training uses the same flow convention through
`models/da3_stitched.py` and `utils/scene_flow_supervision.py` in Plan4D.
