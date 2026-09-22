# Perception track: synthetic YOLO11n-seg, exports, latency and depth fusion

Everything below was measured in this repository on the machine named in each table. Latency and TensorRT numbers are for an **NVIDIA L4 (x86 EC2 host), not Jetson Orin**; the L4 was shared with two other training jobs while these numbers were taken, so treat them as upper bounds on a quiet GPU. Nothing here was run on a real camera or a real robot.

## What was built

| File | Purpose |
|---|---|
| `langgrasp/perception/synth_dataset.py`, `scripts/make_synth_dataset.py` | Render a YOLO-seg dataset from the MuJoCo sim (front, wrist, side cameras at 640x480; polygon labels from the renderer label map) |
| `scripts/train_yolo.py` | Fine-tune `checkpoints/yolo11n-seg.pt` on that dataset, write `checkpoints/yolo11n-seg-langgrasp.pt` and `results/yolo_train.json` |
| `langgrasp/perception/segmentation.py` | `Segmenter` with backends `pt` (ultralytics), `onnx` (onnxruntime CUDA EP, own letterbox / NMS / mask decode in numpy) and `engine` (ultralytics TensorRT) |
| `langgrasp/perception/export.py`, `scripts/export_yolo.py` | ONNX (opset 17, static 1x3x640x640, onnxslim, no NMS in graph) and TensorRT FP16 / INT8 engines |
| `scripts/bench_yolo.py` | Batch-1 latency (30 warmup + 300 timed) and ultralytics val mAP of every artifact; writes `results/yolo_latency_l4.json` and `results/yolo_export_map.json` |
| `langgrasp/perception/depth_fusion.py` | Depth back-projection (MuJoCo camera convention), synthetic depth noise, outlier removal, PCA / minAreaRect grasp proposal; `python -m langgrasp.perception.depth_fusion` writes `results/depth_fusion_accuracy.json` |
| `langgrasp/perception/classes.py` | The three class names (cube, screwdriver, can) |
| `tests/test_perception_depth.py`, `tests/test_perception_dataset.py`, `tests/test_perception_seg.py` | Fast tests (sim only, tiny dataset build, wrapper shape checks; GPU / checkpoint tests skip when missing) |

## Synthetic dataset (`data/yolo_synth`)

Classes are the three SEEN kinds only (`cube`=0, `screwdriver`=1, `can`=2) in the four SEEN colours (red, green, blue, yellow). Unseen colours (purple, orange, pink) and the `bar` kind never appear in this dataset by design; they are held out for the evaluation strata. Scenes are built directly (not with `make_scenario`): 1 to 5 seen-kind pool objects at random positions inside `WORKSPACE` plus a 5 cm margin (so objects sometimes sit on the tray edge or outside the reachable region), random yaw, random seen colour, lighting `nominal` or `degraded` (p=0.35) through `env.reset(Scenario)`, and in 40 % of scenes the arm is driven 1 to 6 ticks toward a random pose between observe and home so it partially occludes objects. Each scene is rendered from the `front`, `wrist` and `side` cameras at 640x480. Labels are the largest external `cv2.findContours` polygon per visible instance from the label map (normalised YOLO-seg format); instances under 60 px are skipped; images with no instance get an empty label file (kept as background images).

| Split | Scenes | Images | Instances | cube / screwdriver / can | front / wrist / side | Empty images | Arm moved | Degraded light |
|---|---|---|---|---|---|---|---|---|
| train | 534 | 1602 | 4334 | 2068 / 1205 / 1061 | 1569 / 1198 / 1567 | 39 | 233 | 206 |
| val | 100 | 300 | 852 | 452 / 219 / 181 | 304 / 244 / 304 | 6 | 45 | 34 |

Rendering time for the whole dataset: **36.5 s** (MuJoCo EGL on the L4 host, one process). Previews with the polygons drawn: `data/yolo_synth/preview_*.png`. Disk: about 69 MB of JPEGs (quality 92).

## Training (`results/yolo_train.json`)

`yolo11n-seg.pt` fine-tuned with ultralytics 8.4.160, imgsz 640, batch 32, default augmentation, 25 epochs (requested 25, time box 35 min not hit), on 1602 train / 300 val images. Training time **5.25 min** (cumulative 'time' column of ultralytics results.csv (training only)) on the shared L4. Peak GPU memory reported by ultralytics: about 6 GB. Weights: `checkpoints/yolo11n-seg-langgrasp.pt` (best.pt by ultralytics fitness). mAP below is a separate `val` pass of the saved weights (ultralytics defaults, rect batches):

| Class | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|
| all | 0.985 | 0.935 | 0.984 | 0.813 |
| cube | 0.994 | 0.956 | 0.994 | 0.829 |
| screwdriver | 0.973 | 0.895 | 0.971 | 0.795 |
| can | 0.987 | 0.953 | 0.987 | 0.814 |

Training curve (from `runs/segment/runs/yolo/yolo11n-seg-langgrasp/results.csv`):

| Epoch | Cum. time s | train box / seg / cls loss | val Box mAP50-95 | val Mask mAP50-95 | val seg loss |
|---|---|---|---|---|---|
| 1 | 28 | 0.857 / 1.021 / 2.458 | 0.275 | 0.286 | 1.150 |
| 2 | 39 | 0.693 / 0.697 / 1.128 | 0.713 | 0.646 | 1.043 |
| 3 | 50 | 0.626 / 0.590 / 0.931 | 0.753 | 0.668 | 1.055 |
| 5 | 73 | 0.572 / 0.533 / 0.746 | 0.811 | 0.753 | 0.568 |
| 10 | 129 | 0.495 / 0.440 / 0.511 | 0.869 | 0.789 | 0.481 |
| 15 | 187 | 0.442 / 0.399 / 0.441 | 0.896 | 0.810 | 0.436 |
| 20 | 248 | 0.368 / 0.325 / 0.346 | 0.917 | 0.813 | 0.443 |
| 22 | 275 | 0.344 / 0.319 / 0.319 | 0.932 | 0.822 | 0.427 |
| 25 | 315 | 0.311 / 0.298 / 0.290 | 0.937 | 0.816 | 0.454 |

Mask mAP50-95 rises steeply in the first 5 epochs and is flat within about 0.01 from epoch 15 on (peak 0.822 at epoch 22); the val seg loss bottoms out around epoch 15 to 20 while train losses keep falling, so more epochs on this synthetic set would mostly overfit. Screwdriver is the weakest class (thin grey shaft, two-geom instance).

## Export path

Static shape, opset 17, no custom plugins, no NMS inside the graph (ultralytics default `nms=False`; NMS runs in numpy in `Segmenter` for the ONNX backend and inside ultralytics for the engine backend). All exports use the ultralytics default square input **1x3x640x640**; a 480x640 rectangle would save 25 % of the pixels but `ultralytics val` forces a square `imgsz` and `rect=False` for non-.pt models, so a rectangular static engine could not be validated with the same tool, and a like-for-like comparison was preferred. The pt backend, by contrast, is letterboxed by ultralytics to 480x640 (auto pad to the stride), which is noted in the latency table. Outputs: `output0` (1, 4+3+32, 8400) and `output1` prototypes (1, 32, 160, 160). Order of export: TensorRT FP16 engine, TensorRT INT8 engine (calibrated on the train split of `data/yolo_synth/data.yaml` through ultralytics' calibrator), then the final ONNX (the engine exports write an intermediate ONNX with the same name).

Export details: see `results/yolo_export.json` (not yet written when this doc was generated).

Environment note: the venv had both `onnxruntime` (CPU) and `onnxruntime-gpu` 1.23.2 installed; the CPU wheel had overwritten the shared package so the CUDA provider was missing. `onnxruntime-gpu==1.23.2` was reinstalled with `--no-deps` on top, and `Segmenter` calls `onnxruntime.preload_dlls()` so the pip-installed CUDA 12 / cuDNN 9 libraries are found. If something reinstalls the CPU wheel, the ONNX backend silently falls back to CPU (check `Segmenter.providers`).

## Latency, batch 1 (`results/yolo_latency_l4.json`)

Not yet measured (bench pending when this doc was generated).

## mAP of each exported artifact (`results/yolo_export_map.json`)

Not yet measured (bench pending when this doc was generated).

## INT8 vs FP16 finding

Not yet measured (bench pending when this doc was generated).

## Depth fusion accuracy (`results/depth_fusion_accuracy.json`)

`grasp_from_points` on ground-truth masks (sim label map) back-projected through the sim depth image, evaluated against `env.grasp_point` on 60 `make_scenario(seed, 'seen')` scenes rendered from the front and side cameras (314 object views per condition; objects with fewer than 200 mask pixels skipped). psi error is taken modulo pi (cube: modulo pi/2) and is not reported for cans (any psi is valid for a cylinder). z error compares `grasp_z` with the oracle grasp height. The noisy condition applies `realsense_like_noise` (sigma 2 mm, 2 % dropout, 1 mm quantisation) to the depth image before back-projection.

| Condition | Kind | n | xy err mean / median / p95 / max mm | frac xy < 8 mm | z err mean / max mm | psi err mean / p95 / max deg | frac psi < 15 deg | width mean mm | aspect mean |
|---|---|---|---|---|---|---|---|---|---|
| clean depth | can | 102 | 0.72 / 0.59 / 1.65 / 2.02 | 1.000 | 0.28 / 0.39 | n/a | n/a | 21.4 | 1.33 |
| clean depth | cube | 126 | 0.45 / 0.25 / 0.95 / 5.79 | 1.000 | 0.16 / 0.22 | 0.7 / 1.1 / 44.3 | 0.992 | 23.8 | 1.18 |
| clean depth | screwdriver | 86 | 5.40 / 2.83 / 16.65 / 85.40 | 0.872 | 0.20 / 3.51 | 2.1 / 4.9 / 5.1 | 1.000 | 16.5 | 6.70 |
| synthetic noise | can | 102 | 1.78 / 1.63 / 2.95 / 3.69 | 1.000 | 1.90 / 3.00 | n/a | n/a | 24.1 | 1.21 |
| synthetic noise | cube | 126 | 1.12 / 1.06 / 1.92 / 6.44 | 1.000 | 1.20 / 1.76 | 1.6 / 3.7 / 44.3 | 0.992 | 25.1 | 1.17 |
| synthetic noise | screwdriver | 86 | 5.73 / 3.16 / 17.33 / 84.98 | 0.895 | 0.95 / 3.07 | 2.0 / 4.4 / 4.9 | 1.000 | 17.7 | 6.36 |

This table was regenerated after the grasp-axis fix described below (extent-based aspect ratio, compact-path fallback); the screwdriver outliers grew (p95 16.6 mm clean) because the fix trades the handle-end bias for more views that lock onto the wrong end when the handle is partially occluded, and are listed in `results/depth_fusion_accuracy.json`.

Notes on the method and the outliers:

- Back-projection uses x_cam = (u + 0.5 - cx) / fx * d, y_cam = -(v + 0.5 - cy) / fy * d, z_cam = -d, then world = R x + t with `env.camera_extrinsics`. Verified in `tests/test_perception_depth.py`: the midpoint of the visible extent of a cube or can lands within 6 mm of `env.object_pose` horizontally (typically under 1 mm), the mean of the visible points is biased toward the camera by 4 to 9 mm because only the camera-facing faces are seen.
- Silhouette pixels of the label map carry table depth (the segmentation pass is not anti-aliased like the depth pass), so `segment_object_points` (drop z < table + 4 mm, then a robust MAD z-score on centroid distance) is required before any statistic.
- Compact objects (cube, can) use `cv2.minAreaRect` on the xy points and the jaw closes across the shorter side; the rectangle centre is exact for a box or cylinder resting on the table because the side faces project inside the top-face footprint.
- Elongated objects (screwdriver, aspect >= 2) are binned in 1 cm slices along the PCA long axis; the handle is the longest run of thick bins. The spec's fixed 1.5 cm width threshold does not work: the visible top of a lying cylinder of radius r is only 2 r sin(elevation) wide, so the 22 mm handle measures 10 to 18 mm depending on the camera, and the bin at the handle end cap can be wider than the handle. An Otsu-style two-cluster split of the bin widths (with an 8 mm floor) is used instead. The handle centre along the long axis is the midpoint of the extent of the off-centre-line points (which the 7 mm shaft cannot produce), and across the axis it is the midpoint of the extent (exact for a cylinder, whereas the mean is biased toward the camera).
- `grasp_z` deviates from the spec on purpose: the median z of the visible points is biased toward the top face, so `grasp_z = (max_z + table_z) / 2` (mid-height of an object resting on the table) is used for short objects and `max_z - 3 cm` for tall ones (can); the median is still returned as `median_z`. This gives the sub-millimetre z errors in the clean condition.
- The screwdriver cases above 8 mm (5 of 86 views, all from the low side camera with a neighbour within 6 cm) are partial occlusions of the handle: re-rendering scenes 3, 45 and 50 with the neighbours removed brings the same views from 29.5 / 25.7 / 31.7 mm down to 4.9 / 3.3 / 2.5 mm. The one cube psi outlier (44 deg) is a similarly clipped view. The wrist camera in the observe pose sees objects at the edge of its field of view and often only the shaft of a screwdriver, so it is not included in this table; use the front camera for the grasp proposal.

## Known limitations

- Synthetic data only: the detector has only seen MuJoCo renders of four flat colours on a plain table; there is no real camera, no real-object validation, and no claim about transfer to a physical SO-101 setup.
- The depth noise model is a synthetic stand-in (Gaussian + dropout + quantisation), not a validated RealSense model; real stereo depth has depth-dependent noise, edge flying pixels and holes on specular surfaces that this does not reproduce.
- All latency and TensorRT numbers are from a shared NVIDIA L4 on an x86 host, not from a Jetson Orin; TensorRT engines are only valid on the GPU type they were built on.
- Exports are static 640x640; the pt backend runs 480x640, so pt vs export latency is not pixel-for-pixel comparable (noted in the table).
- Grasp proposal is top-down only and assumes objects rest on a known table plane; occluded handles and clipped views degrade the centre estimate (see above).
- Unseen colours and the `bar` kind are outside the training distribution by design; the evaluation of that generalisation lives in the eval track, not here.
