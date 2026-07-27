"""
Phase 1b pipeline — live USB webcam on Jetson.

Node graph:
  v4l2_camera   →  /image_raw, /camera_info
  depth_anything_node  →  /depth/image, /depth/camera_info
  rgbd_sync     →  /rgbd_image           (packs rgb + depth + info into one msg)
  rgbd_odometry →  /odom
  rtabmap       →  /cloud_map  (+ loop closure)
  rtabmap_viz   →  GUI viewer            (opt-in, see `viz` arg)

Why the composable container
----------------------------
Measured 2026-07-27: camera alone does 27 Hz and the depth node's own compute
ceiling is 32 fps, but the assembled pipeline ran at 7-10 Hz. The gap was
message transport, not compute — /image_raw is 1.22 MB per frame and
/depth/image is 1.63 MB, and each separate process meant another full
serialize → loopback → deserialize of both.

Two fixes are baked in here:
  * rgbd_sync collapses rgb + depth + camera_info into ONE RGBDImage topic, so
    odometry and rtabmap subscribe once instead of twice each.
  * everything C++ runs in one container with intra-process comms, which turns
    those hops into pointer passing with no serialization at all.

depth_anything_node stays a separate process — rclpy has no zero-copy
intra-process support, so it cannot benefit from the container. /image_raw out
and /depth/image back are the only real IPC hops left.

Large messages between processes (affects viz:=true)
----------------------------------------------------
This box drops big inter-process messages. Measured 2026-07-27 with an external
subscriber, while in-container subscribers saw ~20 Hz of the same topic:

    /odom        (small)   8 msgs / 12 s   ok
    /depth/image (815 KB)  ~20 Hz          ok
    /rgbd_image  (~2 MB)   2 msgs / 12 s   starved

Cause is net.core.rmem_max = 212992 (208 KB, the Linux default) — far too small
for a 2 MB RGBDImage. It is also most of why the pre-container pipeline sat at
7-10 Hz: every one of those 1.2-1.6 MB image messages was crossing a process
boundary through that same undersized socket buffer.

The container sidesteps it for mapping (intra-process = pointer passing, never
serialized), but rtabmap_viz is a GUI and cannot be composed. To use viz:=true:

    sudo sysctl -w net.core.rmem_max=16777216
    # persist it:
    echo 'net.core.rmem_max=16777216' | sudo tee /etc/sysctl.d/60-ros2.conf

Camera calibration:
  Without a calibration file v4l2_camera publishes zeroed intrinsics and the
  3D reconstruction will be wrong. Pass camera_info_url.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    # ── tuneable args ──────────────────────────────────────────────────────────
    engine_path     = LaunchConfiguration("engine_path")
    camera_device   = LaunchConfiguration("camera_device")
    camera_info_url = LaunchConfiguration("camera_info_url")
    frame_id        = LaunchConfiguration("frame_id")
    depth_encoding  = LaunchConfiguration("depth_encoding")
    viz             = LaunchConfiguration("viz")

    # Shared by every rtabmap-family node. approx_sync is required because the
    # depth node adds latency, so depth stamps never match rgb stamps exactly.
    sync_params = {
        "frame_id":        frame_id,
        "subscribe_rgbd":  True,
        "approx_sync":     True,
        "sync_queue_size": 5,
    }

    return LaunchDescription([
        DeclareLaunchArgument(
            "engine_path",
            default_value=str(
                Path.home() / "Depth-Anything-V2" /
                "depth_anything_v2_metric_vits_364_fp16.plan"
            ),
            description="Path to the metric TRT FP16 engine (.plan file)",
        ),
        DeclareLaunchArgument(
            "camera_device",
            default_value="/dev/video0",
            description="V4L2 device node for the USB webcam",
        ),
        DeclareLaunchArgument(
            "camera_info_url",
            default_value="",
            description="file:///path/to/calibration.yaml  (leave empty to use camera default)",
        ),
        DeclareLaunchArgument(
            "frame_id",
            default_value="camera",
            description="Camera frame ID — must match camera_info and TF tree",
        ),
        DeclareLaunchArgument(
            "depth_encoding",
            default_value="16UC1",
            description="16UC1 (millimetres, half the bytes) or 32FC1 (metres). "
                        "16UC1 is the ROS-standard depth encoding and quantizes "
                        "at 1 mm — far below the depth network's own error.",
        ),
        DeclareLaunchArgument(
            "viz",
            default_value="false",
            description="Launch rtabmap_viz. REQUIRES a UDP buffer bump first, "
                        "see the note below, or it will starve. Mapping itself "
                        "is unaffected — odometry and rtabmap live inside the "
                        "container and never touch the wire.",
        ),

        # ── camera (separate process — see note) ───────────────────────────────
        # Deliberately NOT in the container. Loading a component into a process
        # where v4l2_camera is already streaming deadlocks: image_transport
        # dlopens its plugins lazily as subscribers appear, and that races the
        # container's own dlopen for the next component. The container wedges
        # forever on "Load Library: librtabmap_sync_plugins.so". Verified
        # 2026-07-27 — RGBDSync loads into a bare container in 2.5 s, and hangs
        # indefinitely behind a streaming camera even with component_container_mt.
        #
        # Nothing is lost: /image_raw has to cross a process boundary to reach
        # the Python depth node regardless, so the camera gains no intra-process
        # benefit from being inside.
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="camera",
            parameters=[{
                "video_device":    camera_device,
                "image_size":      [848, 480],
                "camera_frame_id": frame_id,
                "camera_info_url": camera_info_url,
            }],
        ),

        # ── depth inference (separate process — Python, no intra-process) ──────
        Node(
            package="depth_anything_node",
            executable="depth_anything_node",
            name="depth_anything_node",
            parameters=[{
                "backend":        "tensorrt",
                "engine_path":    engine_path,
                "input_size":     364,
                "depth_encoding": depth_encoding,
            }],
        ),

        # ── everything C++ in one intra-process container ──────────────────────
        ComposableNodeContainer(
            name="pathfinder_container",
            namespace="",
            package="rclcpp_components",
            # _mt is mandatory, not a preference. The single-threaded
            # component_container hangs here: v4l2_camera's capture loop
            # occupies the lone executor thread, so the LoadNode service call
            # for the next component never gets serviced and the container
            # sits forever on "Load Library: librtabmap_sync_plugins.so".
            executable="component_container_mt",
            output="screen",
            composable_node_descriptions=[
                # Packs rgb + depth + camera_info into a single RGBDImage msg.
                # Assumes depth is registered to rgb and the same size — which
                # it is, the backend resizes the network output back to the
                # frame resolution.
                ComposableNode(
                    package="rtabmap_sync",
                    plugin="rtabmap_sync::RGBDSync",
                    name="rgbd_sync",
                    parameters=[{
                        "approx_sync":     True,
                        "sync_queue_size": 5,
                    }],
                    remappings=[
                        ("rgb/image",       "/image_raw"),
                        ("rgb/camera_info", "/camera_info"),
                        ("depth/image",     "/depth/image"),
                    ],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),

                # ── visual odometry ────────────────────────────────────────────
                # Phase 2: replace this with SPOT body odometry.
                ComposableNode(
                    package="rtabmap_odom",
                    plugin="rtabmap_odom::RGBDOdometry",
                    name="rgbd_odometry",
                    parameters=[sync_params],
                    remappings=[("rgbd_image", "/rgbd_image")],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),

                # ── RGB-D SLAM ─────────────────────────────────────────────────
                ComposableNode(
                    package="rtabmap_slam",
                    plugin="rtabmap_slam::CoreWrapper",
                    name="rtabmap",
                    parameters=[sync_params],
                    remappings=[("rgbd_image", "/rgbd_image")],
                    extra_arguments=[{"use_intra_process_comms": True}],
                ),
            ],
        ),

        # ── viewer (opt-in) ────────────────────────────────────────────────────
        # Display is in feet (pipeline is metres; rtabmap_viz has no built-in
        # unit switch — apply x3.28084 in the point cloud display settings).
        Node(
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            condition=IfCondition(viz),
            parameters=[sync_params],
            remappings=[("rgbd_image", "/rgbd_image")],
        ),
    ])
