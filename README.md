# Pathfinder

**Monocular 3D mapping for Boston Dynamics SPOT.** A single RGB camera plus on-device
machine learning turns SPOT into a mobile 3D scanner: as the robot walks, a Jetson Orin
Nano runs a depth-estimation network on each camera frame and builds a dense, colored
3D reconstruction of the environment in real time — no LiDAR, no depth sensor, no cloud.

The interesting (and risky) part: a single camera physically *cannot* measure depth.
Pathfinder infers it with a neural network ([Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2))
running on the edge, then recovers real-world scale from the robot's odometry.

> **New to the concepts?** [LEARNING.md](LEARNING.md) explains everything from first
> principles — cameras, neural nets, vision transformers, TensorRT, ROS 2, and SLAM.

---

## How it works

```
USB camera ──▶ Depth Anything V2 ──▶ visual/body odometry ──▶ RTAB-Map ──▶ colored 3D map
 (RGB frame)    (TensorRT FP16,       (camera pose)            (SLAM +       (rtabmap_viz)
                 metric depth)                                  loop closure)
```

Every stage is a ROS 2 node connected by topics. The depth network outputs metric depth
(actual metres), which RTAB-Map fuses with pose estimates to assemble a globally
consistent point cloud colored with the camera's RGB.

The full node graph and topic contract live in [ARCHITECTURE.md](ARCHITECTURE.md).
The phased project plan and rationale live in [BRIEFING.md](BRIEFING.md).

---

## In action

`rtabmap_viz` running live during a Phase 1b handheld walk on the Jetson — Logitech BRIO
webcam → Depth Anything V2 (TensorRT FP16) → RTAB-Map. Left panels show loop-closure
detection (top) and the live depth colormap from the odometry node (bottom); the right
panel is the dense colored 3D point cloud growing in real time.

<p align="center">
  <img src="documentation/images/Screenshot%20from%202026-06-15%2014-21-06.png" width="49%">
  <img src="documentation/images/Screenshot%20from%202026-06-15%2014-43-36.png" width="49%">
</p>

The second frame is later in the same walk: the cyan graph in the 3D Map panel is a
detected loop closure (the robot recognized a revisited area), which RTAB-Map uses to
correct accumulated drift as the map keeps growing.

---

## Hardware

| Component | Part | Notes |
|---|---|---|
| Compute  | **Jetson Orin Nano (8GB)** | JetPack 7.2 — Ubuntu 24.04, CUDA 13.2, TensorRT 10.16 |
| Camera   | Any **UVC USB webcam** | Currently a Logitech BRIO; target is the Arducam IMX462 (mono, night-vision). The camera node is config, not code — any UVC device drops in. |
| Robot    | Boston Dynamics SPOT | Phase 2+ — not required for the standalone mapping pipeline |

**Hard constraints (see [CLAUDE.md](CLAUDE.md) for the full list):**
- 8GB shared RAM → depth model is the **Small** variant only.
- **FP16 is mandatory** for usable framerate (unlocks the GPU's Tensor Cores).
- **TensorRT engines must be built on the Jetson** — they're specific to the exact GPU + TRT version.
- Plug the camera into a **USB-A** port (the Jetson's USB-C is device-mode only).

---

## Repository layout

```
Pathfinder/                              ← project root + colcon workspace
│
├── README.md                            ← this file
├── CLAUDE.md                            ← working notes, status, hard rules, gotchas
├── BRIEFING.md                          ← durable plan: phases, architecture, risks
├── ARCHITECTURE.md                      ← current node graph + topic contract
├── LEARNING.md                          ← first-principles guide to every concept used
├── requirements.txt                     ← project Python deps (installed into .venv)
│
├── scripts/                             ← one-off setup / runbook scripts (not shipped code)
│   ├── phase1b_setup.sh                 # installs ROS 2 Jazzy, creates .venv, wires .bashrc
│   ├── phase1b_export_metric.py         # metric DAv2-Small → ONNX (then build TRT engine)
│   └── phase1b_calibrate.py             # standalone OpenCV camera calibration
│
├── src/depth_anything_node/             ← the custom ROS 2 package
│   ├── depth_anything_node/
│   │   ├── node.py                      # ROS node: subscribes RGB, publishes metric depth
│   │   └── backends.py                  # TorchBackend | TensorRTBackend (selectable)
│   ├── launch/pipeline.launch.py        # full pipeline: camera → depth → odometry → SLAM → viewer
│   ├── package.xml / setup.py / setup.cfg
│   └── resource/
│
├── .venv/                               ← project venv (--system-site-packages; gitignored)
├── build/ install/ log/                 ← colcon outputs (gitignored)
└── models/                              ← TRT engines + checkpoints (created as needed)
```

---

## Setup (Jetson Orin Nano)

### 1. Install ROS 2 + create the venv

```bash
cd ~/Documents/Projects/Pathfinder
bash scripts/phase1b_setup.sh
```

This installs ROS 2 Jazzy (`ros-jazzy-desktop`, `rtabmap-ros`, `v4l2-camera`,
`cv-bridge`, `camera-calibration`), creates `.venv/` with `--system-site-packages`
(so it sees the system's ROS 2 and TensorRT), installs `requirements.txt`, and adds
the source lines to `~/.bashrc`.

**Verify** (in a new terminal):
```bash
ros2 pkg list | grep rtabmap
python3 -c "from cuda.bindings import runtime; print('cuda.bindings OK')"
```

> ⚠️ In **VS Code terminals**, `~/.bashrc` may not source (interactive-shell guard).
> If `ros2` isn't found, run `source /opt/ros/jazzy/setup.bash` directly, or move the
> source lines into `~/.profile`.

### 2. Build the metric TensorRT engine

The engine must be built **on the Jetson**. First authenticate with Hugging Face
(`hf auth login`), then export the model to ONNX:

```bash
cd ~/Depth-Anything-V2
~/Documents/Projects/Pathfinder/.venv/bin/python \
  ~/Documents/Projects/Pathfinder/scripts/phase1b_export_metric.py
```

This downloads the metric DAv2-Small model via `transformers` and exports it to ONNX,
then **prints the `trtexec` command** to build the FP16 engine. Run that command
(takes 3–10 min). Result on the Orin Nano: **~70 fps, ~14 ms latency**.

Engine path: `~/Depth-Anything-V2/depth_anything_v2_metric_vits_364_fp16.plan`

### 3. Build the ROS 2 package

```bash
cd ~/Documents/Projects/Pathfinder
colcon build && source install/setup.bash
ros2 pkg list | grep depth_anything_node     # verify
```

### 4. Calibrate the camera

The ROS calibration tool crashes with `v4l2_camera` (no `set_camera_info` service),
so use the standalone OpenCV script. Print or display an 8×6 checkerboard:

```bash
~/Documents/Projects/Pathfinder/.venv/bin/python scripts/phase1b_calibrate.py
# SPACE/c = capture a frame, Enter = calibrate (after ≥15 frames), q = quit
```

Saves intrinsics to `~/.ros/camera_info/camera.yaml`.

---

## Running the pipeline

```bash
cd ~/Documents/Projects/Pathfinder
source install/setup.bash
ros2 launch depth_anything_node pipeline.launch.py \
  camera_info_url:=file:///home/$USER/.ros/camera_info/camera.yaml
```

`rtabmap_viz` opens. Walk the camera slowly around the room — a growing colored point
cloud appears in the viewer.

**Launch arguments** (all optional, with sensible defaults):

| Argument | Default | Purpose |
|---|---|---|
| `engine_path` | `~/Depth-Anything-V2/...vits_364_fp16.plan` | TRT engine to load |
| `camera_device` | `/dev/video0` | V4L2 device node |
| `camera_info_url` | *(empty)* | `file://` path to calibration YAML |
| `frame_id` | `camera` | Coordinate frame — must match the camera_info `camera_name` |

---

## Project status

| Phase | What | Status |
|---|---|---|
| **0**  | TensorRT depth benchmark on Jetson | ✅ Done — 40.7 fps, monocular approach confirmed viable |
| **1a** | Pipeline correctness on laptop (WSL2, PyTorch) | ✅ Done — full pipeline proven on TUM dataset |
| **1b** | Migrate to Jetson (TensorRT + live webcam) | ✅ Done — handheld walk → live colored point cloud |
| **2**  | SPOT integration (body odometry, TF tree) | ⏸ Next (on hold) |
| **3**  | Fused mapping on SPOT — **core deliverable** | Planned |
| **4**  | Quality + optimization | Planned |
| **5**  | Application layer (change detection / nav / inspection) | Planned |

Current status is tracked authoritatively in [CLAUDE.md](CLAUDE.md).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `getting transform camera_optical_frame -> camera` | `frame_id` mismatch. The launch `frame_id` must match the `camera_name` in the calibration YAML (both should be `camera`). |
| Black camera frames | OpenCV on JetPack uses the GStreamer backend. Don't force `cv2.CAP_V4L2` — it bypasses GStreamer. |
| `trtexec: Static model does not take explicit shapes` | The ONNX has static shapes baked in. Drop `--minShapes/--optShapes/--maxShapes`. |
| Hugging Face 401 / auth error | Run `hf auth login` (not the deprecated `huggingface-cli login`). |
| `ros2: command not found` in VS Code | `~/.bashrc` not sourced. Run `source /opt/ros/jazzy/setup.bash`. |
| Camera shows IR / wrong stream | On the BRIO, `/dev/video0` is RGB; `/dev/video2` is the IR sensor. |
| `Failed getting value for control ...: Permission denied` | Non-fatal — a proprietary UVC extension control. The camera streams fine. |
| Map silently never builds | RGB and depth timestamps not matching. `approx_sync: True` is set; if still failing, the depth node may be too slow for the frame rate. |

More detail and the reasoning behind each fix is in [LEARNING.md](LEARNING.md) and
the gotchas sections of [CLAUDE.md](CLAUDE.md).

---

## Documentation map

- **[CLAUDE.md](CLAUDE.md)** — how to work on the project, environment, hard rules, current status, gotchas.
- **[BRIEFING.md](BRIEFING.md)** — the durable plan: what Pathfinder is, the architecture, the phases, the risks.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the current ROS 2 node graph and topic contract.
- **[LEARNING.md](LEARNING.md)** — a teach-yourself guide to every concept the project touches.

## License

Apache-2.0 (per the ROS package manifest).
