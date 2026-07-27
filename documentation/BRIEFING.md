# Pathfinder — Project Briefing

**Status:** see Current Status in [CLAUDE.md](CLAUDE.md) (single source of truth for status)
**Owner:** Orlando
**Approach:** monocular depth estimation → dense reconstruction
**Last updated:** 2026-06-15 — Phase 1b complete; Phase 2 next (SPOT integration). Node graph in [ARCHITECTURE.md](ARCHITECTURE.md)

---

## What this is

Pathfinder turns a Boston Dynamics SPOT robot into a mobile 3D mapping platform using a **single monocular camera** plus on-board machine learning. A Jetson Orin Nano and an Arducam IMX462 USB camera ride on SPOT's payload. As SPOT walks, the Jetson runs a monocular depth-estimation network on each frame, fuses the result with SPOT's body odometry to recover real-world scale, and builds a dense colored 3D reconstruction of the environment — a navigable map of what the robot is seeing.

This is deliberately the ML-heavy version of the project. The camera can't measure depth, so depth is *inferred* by a neural network on the edge device. That's the whole point and the whole risk.

All processing happens on the robot — no cloud dependency. The camera's night-vision capability makes after-dark and low-light inspection (tunnels, underpasses) a natural fit.

The system has two halves: a **back-end** (the on-Jetson pipeline that produces the reconstruction) and a **front-end** (a visualization client that shows the 3D map being built, colored with the camera's RGB).

---

## Why this architecture (and the honest catch)

A monocular camera cannot recover metric depth or dense geometry from a single frame — that's geometry, not a software gap. Two consequences drive the whole design:

1. **Depth is estimated, not measured.** A depth network (Depth Anything V2-Small) predicts per-pixel depth from each RGB frame. This output is *relative*, not metric, and it is noticeably noisier than a real depth sensor. The reconstruction will not look as clean as a RealSense-based map. Set expectations accordingly.

2. **Scale comes from SPOT, not the camera.** Relative depth has no real-world units. SPOT's body odometry rescales the relative depth into metric depth — this is a validated technique (monocular visual-inertial / odometric rescaling) and it is *load-bearing*, not optional. Without it, the map has no consistent scale and drifts.

If either of those turns out to be unworkable on the hardware, the fallback is to switch to a stereo/depth camera and drop the depth network entirely. Phase 0 is designed to find that out before any time is sunk into SPOT integration.

---

## Hardware

| Component | Part | Notes |
|---|---|---|
| Robot | Boston Dynamics SPOT | CORE payload running Ubuntu |
| Compute | Jetson Orin Nano (8GB) | JetPack 7.2 — see version risk below |
| Camera | Arducam IMX462 Day/Night USB | Monocular RGB, USB 2.0, 1080p@30, rolling shutter, 95° FOV, 940nm IR night vision to ~3m |

Hard limits this imposes:
- **8GB shared RAM caps the depth model to the Small variant.** Base/Large do not fit on Orin.
- **USB 2.0 bandwidth** caps practical resolution/framerate — don't expect full 1080p30 into the pipeline.
- **Rolling shutter + no IMU** means motion artifacts under SPOT's gait; mono is more sensitive to this than stereo.

---

## Architecture (target)

```
BACK-END (on Jetson, on SPOT)
─────────────────────────────
Arducam IMX462  (mono RGB frames, USB 2.0)
        │
        ▼
Depth Anything V2-Small   (TensorRT FP16, on Jetson GPU)
        │  relative depth, per frame
        ▼
Scale recovery  ◄──── SPOT body odometry (spot_ros2)   # rescales relative → metric, supplies pose
        │  metric pseudo-depth  +  pose
        ▼
Dense mapper   (nvblox  OR  RTAB-Map)
        │  metric point cloud / mesh (with RGB per point)
        ├──► saved map → disk (per mission)
        │
        ▼  (stream over ROS 2 / network)
FRONT-END (viewer — laptop or browser)
─────────────────────────────
   render the reconstruction (RGB / camera color)
```

Design decisions:
- **Depth model:** Depth Anything V2-Small, converted to TensorRT FP16. Small is the only size that fits; FP16 is required for usable framerate.
- **Pose + scale:** SPOT odometry, not visual odometry. Solves both the gait-bounce problem and the metric-scale problem in one move.
- **Mapper:** nvblox is the GPU-native choice and takes depth+color+pose directly, but carries the JetPack/version risk. RTAB-Map is the less-version-coupled fallback. Decide in Phase 1 based on what actually builds.
- **Front-end:** the reconstruction is shown colored by the camera's RGB. RViz/Foxglove handle this for free early; a custom web viewer (e.g. three.js) is the polished, shareable version and its own workstream.

---

## Phases

### Phase 0 — Bench foundation + depth benchmark (MAKE-OR-BREAK) — ✅ DONE
Off the robot entirely. Get the camera streaming, ROS 2 up (native on the Jetson), GPU usable. **Then the decisive step:** convert Depth Anything V2-Small to TensorRT FP16 and measure real framerate on *this* Orin Nano at a candidate input size.
**Exit:** depth network running at a usable rate (target: ≥~10–15 fps at a low input resolution). If it can't hit that, the monocular approach is not viable — stop here and reconsider the sensor. This is the cheapest possible place to kill the idea.

**Result (PASSED):** Depth Anything V2-Small, TensorRT FP16, on the Orin Nano —
at 364×364: ~8.99 ms preprocess + ~15.59 ms inference = ~24.58 ms → **40.7 fps
realistic** (synthetic frames). ~4× the 10 fps floor. Monocular approach confirmed
viable with margin — and with this much headroom over the ≥10–15 fps bar, there's
room to run a lower Jetson power mode and still clear it. Scripts/engines in `phase0_benchmark/`.
**Gotchas recorded (durable):**
- ONNX export needs `pip install onnxscript` on current PyTorch.
- A venv must set `include-system-site-packages = true` to see JetPack's system TensorRT.
- cuda-python moved its bindings: `from cuda.bindings import runtime as cudart`.
- Input size must be a multiple of 14 (DINOv2 patch size).
- Build the TensorRT engine ON the Jetson — engines are GPU/TRT-version specific.
**Not yet done (non-blocking):** confirm fps with a live camera (`--camera 0`);
benchmark 308 (will be faster). Neither changes the verdict.

### Phase 1a — Laptop PoC, pipeline correctness (WSL2, PyTorch depth) — ✅ DONE

**Result (PASSED, 2026-06-11):** Full pipeline running on WSL2 laptop. TUM fr1/desk
sequence replayed → depth_anything_node → rgbd_odometry → rtabmap → rtabmap_viz.
Camera trajectory and growing dense point cloud confirmed in viewer. Pipeline logic
proven correct. ROS 2 packages in `~/pathfinder/src/depth_anything_node/`.

**Gotchas recorded (durable):**
- Metric Depth Anything V2-Small runs at ~1.567 Hz on RTX 2000 Ada (avg 638ms/frame).
  Frame publisher must run at ≤1 fps — faster rates cause approx_sync timestamp
  misses and the map silently fails to build.
- `ament_python` hardwires `/usr/bin/python3` into entry point scripts regardless of
  venv state. Fix: add venv site-packages to PYTHONPATH in `~/.bashrc`.
- HuggingFace pipeline `"depth"` key returns a normalized uint8 image (0–255), not
  metric depth. Use `result["predicted_depth"].squeeze().cpu().numpy()` for metres.
- `pathfinder_venv` must be created with `--system-site-packages` so rclpy and other
  ROS Python packages are visible inside it.

Differences from Jetson build, by design:
- **Depth backend = PyTorch**, not TensorRT (correctness PoC, speed already proven in Phase 0).
- **Frame source = TUM RGB-D recorded sequence**, not a live camera (USB cams don't pass cleanly into WSL2).

**Exit criteria met:** recognizable, coherent reconstruction in rtabmap_viz. Pipeline logic proven.

### Phase 1b — Migrate to Jetson (TensorRT, live camera) — ✅ DONE

**Result (PASSED, 2026-06-15):** Full pipeline running on Jetson Orin Nano. Logitech BRIO (848×480) → v4l2_camera → depth_anything_node (TRT FP16, **70.3 fps, 14.16 ms**) → rgbd_odometry → rtabmap → rtabmap_viz. Handheld walk produced a growing colored point cloud. Exit criteria met.

**Durable gotchas:**
- HuggingFace metric model is safetensors-only on Hub; use `from_pretrained()` via transformers, not raw `.pth` download.
- ONNX static shapes baked in → `trtexec` must NOT use `--minShapes/--optShapes/--maxShapes`.
- OpenCV on JetPack uses GStreamer backend; `CAP_V4L2` gives black frames.
- ros-jazzy-camera-calibration crashes (v4l2_camera has no `set_camera_info` service). Standalone OpenCV calibration script used instead.
- YAML camera_info must use inline sequences (`[v1, v2, ...]`) — yaml-cpp rejects block sequences.
- v4l2_camera publishes `frame_id` from YAML `camera_name`. All rtabmap nodes must use the same frame name (`camera`). Mismatch = TF lookup failure.
- USB-C on Jetson Orin Nano is device-mode only — cameras go into USB-A.

**Throughput rework (2026-07-27): 7–10 Hz → 20.9 Hz.** The as-shipped pipeline ran far below its parts: the camera alone did 27 Hz and the depth node's compute ceiling was 32 fps, but five separate processes serializing 1.2–1.6 MB image messages to three or four subscribers each cost ~9.8 MB of copying per frame. `rgbd_odometry`, `rtabmap` and a new `rtabmap_sync::RGBDSync` now run composed in one `component_container_mt` with intra-process comms; depth moved to `16UC1` millimetres (half the bytes of `32FC1`); `rtabmap_viz` became opt-in. No change to the model, the engine, or the camera.

This matters beyond Phase 1b: **the plumbing, not the network, is the thing that will constrain Phase 3.** SPOT adds more subscribers and more topics to the same graph, and the same serialization cost applies to each. Budget for composition when designing the Phase 2/3 node graph rather than retrofitting it.

It also surfaced a latent environment bug worth carrying forward: Ubuntu's default socket buffers (208 KB across all four `net.core.*mem_*` values) are too small for uncompressed image topics, and DDS drops the excess **silently**. It went unnoticed through all of Phase 1b because at 7–10 Hz the traffic still fit. See documentation/CLAUDE.md for the fix — and note that raising `rmem_max` alone does nothing, which is the obvious wrong turn.

### Phase 2 — SPOT integration (no mapping yet)
Mount Jetson + camera. Bring up spot_ros2. Get SPOT body odometry + state into ROS 2. Build the TF tree (SPOT body → camera). Stand up the scale-recovery step that ties odometry to the depth output.
**Exit:** SPOT live odometry and camera frames time-synced in one ROS graph, and relative depth getting rescaled to metric using odometry.

### Phase 3 — Fused mapping on SPOT (CORE DELIVERABLE)
SPOT odometry drives both pose and metric scale; estimated depth supplies geometry. Walk a route under teleop.
**Exit:** SPOT walks a corridor → usable, metrically-scaled dense map. **This is minimum-viable Pathfinder.** Everything after is refinement.

### Phase 4 — Quality + optimization
Address the failure modes the first working map exposes:
- **Per-frame depth noise** — depth is computed independently per frame, so surfaces jitter and come out fuzzy/thick. Fix: temporal filtering of the depth stream; let the mapper's volumetric integration average repeated observations of the same point.
- **Boundary artifacts** — networks misjudge object edges, scattering "flying pixels" into the space between foreground and background. Fix: discard low-confidence pixels at depth discontinuities.
- **Drift** — small odometry + monocular-scale errors accumulate over a long walk. Fix: RTAB-Map loop closure (recognize a revisited place, correct the accumulated error); periodically re-anchor scale to SPOT odometry.
- **Motion artifacts** — rolling shutter + gait warp frames during quick motion. Fix: compensate for, or drop, frames captured during high angular velocity.
- **Throughput** — if the full pipeline can't keep pace on the Jetson once everything runs together: INT8 over FP16, lower input resolution, move preprocessing onto the GPU.
- **Validation** — compare reconstructed dimensions against known real-world measurements, turning "looks right" into an error figure.

Note: cuVSLAM is NOT available here (stereo-only) — this phase squeezes the mono pipeline, it doesn't swap in the Isaac VO stack.
**Exit:** measurably cleaner / more stable maps than Phase 3.

### Phase 5 — Application layer (optional, the differentiator)
With a working map, one of these turns Pathfinder into a capability:
- **Map reuse / change detection** — save an inspection mission's map, relocalize SPOT in it on a later visit, and difference the two scans to highlight what changed (new debris, settlement, a fresh defect). The most direct inspection fit.
- **Light autonomous navigation** — convert the reconstruction into a costmap and let SPOT traverse a mapped area autonomously (Nav2). The most technically ambitious option.
- **Inspection mode** — run a defect/asset detector alongside the mapper and project its 2D detections into the 3D map using depth + pose, so each finding is spatially located. The most report-ready artifact; leans on the night-vision strength for low-light inspection.

**Exit:** defined by which is chosen.

### Front-end (parallel workstream, not a single phase)
The viewer matures alongside the back-end rather than waiting for a dedicated phase:
- **Phase 1+:** use RViz or Foxglove as the viewer — both render the RGB point cloud out of the box, which is the debugging view you need anyway.
- **Phase 3+:** build the custom web front-end (e.g. three.js/WebGL) — the polished, shareable "show the mapping happening" deliverable.
**Exit (custom viewer):** live-updating 3D map in the browser.

---

## Known risks

1. **Depth framerate on 8GB Orin Nano.** The make-or-break unknown. Most strong fps numbers in the literature are AGX Orin, which is far beefier. Mitigation: benchmark in Phase 0 before anything else.
2. **Metric scale.** Mono depth is relative. Mitigated by odometric rescaling from SPOT — load-bearing, validated, but must actually be implemented (Phase 2).
3. **Reconstruction quality.** Estimated depth is noisy; maps will be visibly rougher than a depth-sensor map. This is inherent to the approach, not a bug. Manage expectations in any demo/pitch.
4. **JetPack 7.2 ahead of Isaac ROS tested matrix.** Applies if using nvblox. Mitigation: RTAB-Map fallback is far less version-coupled and carries the project to a working demo regardless.
5. **GPU passthrough into Docker.** A classic Jetson failure point — but development is native, so this only applies later if/when containerizing for deployment.
6. **Rolling shutter + gait artifacts.** Mono is sensitive. Partly addressed by odometry fusion and Phase 4 temporal filtering.
7. **Scope creep into Phase 4/5.** Don't let refinement block having a working demo at end of Phase 3.

---

## Next actions

Phase 0 is complete (see its result above). The **active phase and current next
actions live in [CLAUDE.md](CLAUDE.md) → Current Status** (single source of truth) — one place
to look, nothing to drift here.

Non-blocking carryovers / future tasks:
- **Live camera check** — any UVC webcam for 1b (the IMX462 isn't on hand yet); real `--camera 0` confirmation happens on the Jetson. IMX462 swaps in later, calibrate-and-go.
- **Metric model engine (Phase 1b)** — re-export the *metric* Depth Anything V2 `.pth` → ONNX → TensorRT FP16 using the `phase0_benchmark/` scripts. The existing engine is the *relative* model. (Phase 1a uses PyTorch, no TensorRT.)
- **Jetson ROS 2 install (Phase 1b)** — decision: **native** (Phase 0 already runs natively; avoids GPU + USB-camera passthrough on JetPack 7.2; matches the native 1a laptop). Containerize later for SPOT deployment or if Isaac/nvblox is adopted.
