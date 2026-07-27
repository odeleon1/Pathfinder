# ARCHITECTURE — Phase 1 (handheld, no SPOT)

Phase 1 builds the depth→reconstruction pipeline before SPOT is involved.
Everything runs in ROS 2 so none of it is throwaway — later phases swap individual
boxes, not the graph.

**Same graph runs in 1a (laptop) and 1b (Jetson). Only two boxes differ:**
- **depth backend:** 1a = PyTorch (laptop GPU); 1b = TensorRT engine (Jetson).
  Behind one ROS interface, chosen by a param — so the graph is unchanged.
- **frame source:** 1a = recorded video/rosbag replay (USB cams don't pass cleanly
  into WSL2); 1b = live USB webcam UVC node (IMX462 once available).
Everything between (depth node interface, odometry, rtabmap, viewer) is identical.
That identity is the point: 1a proves the logic, 1b is migration + two swaps.

## Node graph

Three OS processes. The dashed box is a single `component_container_mt` — the nodes
inside it talk by passing pointers, not by serializing messages onto the network stack.

```
  process 1                      process 2
┌─────────────────┐  /image_raw (rgb8)        ┌──────────────────────┐
│  camera node    │ ─────────────────────────►│  depth_anything_node │
│ (v4l2_camera)   │  /camera_info             │   (custom, Python)   │
│                 │ ──────────────┐           │  - wraps TRT engine  │
└─────────────────┘               │           │  - RGB in, depth out │
                                  │           └──────────┬───────────┘
                                  │   /depth/image (16UC1, millimetres)
                                  ▼                      ▼
  ╭┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╮
  ┊  process 3 — component_container_mt (intra-process comms)   ┊
  ┊                                                             ┊
  ┊    ┌───────────────────────────────────────────────┐        ┊
  ┊    │  rgbd_sync  (rtabmap_sync::RGBDSync)          │        ┊
  ┊    │   rgb + depth + camera_info → ONE message     │        ┊
  ┊    └──────────────────────┬────────────────────────┘        ┊
  ┊                           │ /rgbd_image                     ┊
  ┊              ┌────────────┴────────────┐                    ┊
  ┊              ▼                         ▼                    ┊
  ┊    ┌──────────────────┐     ┌────────────────────────────┐  ┊
  ┊    │  rgbd_odometry   │     │  rtabmap  (RGB-D SLAM)     │  ┊
  ┊    │  → /odom         │────►│  → /cloud_map, loop closure│  ┊
  ┊    │  (P2: SPOT odom) │/odom└────────────────────────────┘  ┊
  ┊    └──────────────────┘                                     ┊
  ╰┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╯
                                  │
                                  ▼  (opt-in: viz:=true)
                    ┌───────────────────────────────────────┐
                    │  rtabmap_viz                          │
                    │   subscribes /image_raw + /depth/image│
                    │   NOT /rgbd_image — see below         │
                    └───────────────────────────────────────┘
```

**Why the container.** Measured 2026-07-27: the camera alone does 27 Hz and the depth
node's compute ceiling is 32 fps, but the pipeline assembled from five separate
processes ran at 7–10 Hz. The cost was serializing 1.2–1.6 MB image messages onto the
loopback network stack once per subscriber. Composing the three SLAM nodes and
collapsing rgb+depth+info into one `RGBDImage` took it to **20.9 Hz**.

**Why the viewer is the odd one out.** `rtabmap_viz` is a GUI and cannot be composed, so
it has to receive messages over the wire — and on this box the image topics do not
survive that trip once they already have subscribers (see *fan-out*, below). The
supported mode is **map-only**: the viewer takes `/mapData` and `/info` and displays the
3D cloud and graph, with no camera or depth preview panels. The launch file's `viz:=true`
is left in place but does not work.

**Fan-out limit.** A large topic serves its *first* subscriber at full rate and starves
every additional one — the constraint is subscriber count, not message size. Measured
2026-07-27: a third subscriber to `/image_raw` (1.22 MB) got 0.20 Hz, while the same data
on a dedicated single-subscriber topic ran at 24 Hz. Compression sidesteps it entirely
(`/image_raw/compressed` is 107 KB and delivers 24.4 Hz). This matters when adding any
new consumer to an existing image topic.

**Containment.** Topics published inside the container mostly do not escape it. `/tf`
does (17.7 Hz externally); `/odom` does not (0.13 Hz), despite being a few hundred bytes.
Nothing outside needs `/odom` today. Phase 2 changes that — SPOT's odometry crosses this
same boundary.

## Why each piece

- **camera node** — pulls frames off the camera (any UVC USB device: a generic
  webcam for interim Phase 1b, the Arducam IMX462 once it's available). Publishes
  raw RGB and the `camera_info` (intrinsics from calibration). Intrinsics must be
  correct or the 3D reconstruction is geometrically wrong — and they're per-camera,
  so the webcam and the IMX462 each need their own calibration.
- **depth_anything_node** — the custom heart of the system. Subscribes to RGB,
  runs the metric Depth Anything V2 TensorRT engine (reusing `TRTRunner` from the
  Phase 0 `benchmark_e2e.py`), publishes a depth image aligned to the RGB frame.
  This is the node that turns a single camera into an RGB-D sensor.
- **rgbd_odometry** — RTAB-Map's visual odometry. In Phase 1 there is no external
  odometry, so this estimates camera motion from the images. **Phase 2 replaces
  this with SPOT body odometry** — that swap is the main Phase 2 change.
- **rtabmap** — RGB-D graph SLAM. Builds the map, does loop closure (recognizing
  revisited places to correct drift), outputs the dense colored point cloud.
- **viewer** — RTAB-Map/RViz displays the dense point cloud colored by the camera's
  RGB. This is the debugging view during Phase 1 and the basis for the later custom
  web viewer. **Display unit: feet.** The pipeline and all ROS topics carry depth in
  metres (ROS convention); the viewer converts to feet at display time only
  (multiply by 3.28084). All internal logic stays in metres.

## Topic contract (the interfaces that matter)

| Topic | Type | Size/frame | Produced by | Consumed by |
|---|---|---|---|---|
| `/image_raw` | sensor_msgs/Image (rgb8) | 1.22 MB | camera | depth_anything_node, rgbd_sync, rtabmap_viz |
| `/camera_info` | sensor_msgs/CameraInfo | tiny | camera | depth_anything_node, rgbd_sync |
| `/depth/image` | sensor_msgs/Image (16UC1, mm) | 815 KB | depth_anything_node | rgbd_sync, rtabmap_viz |
| `/depth/camera_info` | sensor_msgs/CameraInfo | tiny | depth_anything_node | rtabmap_viz |
| `/rgbd_image` | rtabmap_msgs/RGBDImage | ~2 MB | rgbd_sync | rgbd_odometry, rtabmap *(in-process only)* |
| `/odom` | nav_msgs/Odometry | tiny | rgbd_odometry (P1) → SPOT (P2) | rtabmap, rtabmap_viz |
| `/cloud_map` | sensor_msgs/PointCloud2 | grows | rtabmap | viewer |

The depth must be **registered** to (aligned with) the RGB frame — same resolution,
same intrinsics. That contract is most of depth_anything_node's job.

**On the depth encoding.** The wire format is `16UC1` millimetres, not `32FC1` metres:
half the bytes on a topic that is bandwidth-bound, and it is the standard ROS depth
encoding that every RGB-D consumer already understands. 1 mm quantization is far below
the depth network's own error, so nothing is lost. `depth_encoding:=32FC1` switches back.
Note this is a *wire* format — everything upstream and downstream still reasons in
metres, per the project convention.

**`/rgbd_image` does not leave the container.** It is listed here for completeness, but
an out-of-process subscriber will not receive it reliably on default socket buffers.
Anything outside the container must use the raw topics.

## What changes in later phases (so Phase 1 doesn't paint us into a corner)

- **Phase 2:** `/odom` comes from SPOT instead of rgbd_odometry; add the TF tree
  (SPOT body → camera). depth_anything_node and rtabmap are unchanged.
- **Phase 4:** optionally swap rtabmap for nvblox; add temporal filtering on the
  depth stream. The node *interfaces* above stay the same.
