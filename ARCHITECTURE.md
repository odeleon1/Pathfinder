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

```
┌─────────────────┐  /image_raw (rgb8)        ┌──────────────────────┐
│  camera node    │ ─────────────────────────►│  depth_anything_node │
│ (v4l2_camera /  │  /camera_info             │   (custom)           │
│  usb_cam)       │ ──────────────┐           │  - wraps TRT engine  │
└─────────────────┘               │           │  - RGB in, depth out │
                                  │           └──────────┬───────────┘
                                  │   /depth/image (32FC1, metres)   │
                                  │   /depth/camera_info             │
                                  ▼                                  ▼
                          ┌───────────────────────────────────────────────┐
                          │  rgbd_odometry  (RTAB-Map visual odometry)      │
                          │   RGB+depth → /odom  (Phase 2: replace w/ SPOT) │
                          └───────────────────────┬─────────────────────────┘
                                                  │ /odom
                                                  ▼
                          ┌───────────────────────────────────────────────┐
                          │  rtabmap  (RGB-D SLAM)                          │
                          │   RGB + depth + odom → map + loop closure       │
                          │   out: /rtabmap/cloud_map (dense point cloud)   │
                          └───────────────────────┬─────────────────────────┘
                                                  │
                                                  ▼
                          ┌───────────────────────────────────────────────┐
                          │  viewer (rtabmap_viz / rviz2)                   │
                          │   shows the dense RGB point cloud               │
                          │   (/rtabmap/cloud_map, colored by camera RGB)   │
                          └───────────────────────────────────────────────┘
```

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

| Topic | Type | Produced by | Consumed by |
|---|---|---|---|
| `/image_raw` | sensor_msgs/Image (rgb8) | camera | depth_anything_node, rgbd_odometry, rtabmap |
| `/camera_info` | sensor_msgs/CameraInfo | camera | depth_anything_node, rtabmap |
| `/depth/image` | sensor_msgs/Image (32FC1, metres) | depth_anything_node | rgbd_odometry, rtabmap |
| `/odom` | nav_msgs/Odometry | rgbd_odometry (P1) → SPOT (P2) | rtabmap |
| `/rtabmap/cloud_map` | sensor_msgs/PointCloud2 | rtabmap | viewer |

The depth must be **registered** to (aligned with) the RGB frame and in **metres**
(32FC1). Getting that contract right is most of the depth_anything_node's job.

## What changes in later phases (so Phase 1 doesn't paint us into a corner)

- **Phase 2:** `/odom` comes from SPOT instead of rgbd_odometry; add the TF tree
  (SPOT body → camera). depth_anything_node and rtabmap are unchanged.
- **Phase 4:** optionally swap rtabmap for nvblox; add temporal filtering on the
  depth stream. The node *interfaces* above stay the same.
