"""
Phase 1b pipeline — live USB webcam on Jetson.

Nodes launched:
  v4l2_camera   →  /image_raw, /camera_info
  depth_anything_node  →  /depth/image, /depth/camera_info
  rgbd_odometry →  /odom
  rtabmap       →  /rtabmap/cloud_map  (+ loop closure)
  rtabmap_viz   →  GUI viewer (feet display, x3.28084)

Camera calibration:
  Run camera_calibration first (see scripts/phase1b_setup.sh notes).
  Without a calibration file, v4l2_camera publishes zeroed intrinsics and
  the 3D reconstruction will be wrong. For a quick smoke test you can use
  the camera_info_url workaround below.
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── tuneable args ──────────────────────────────────────────────────────────
    engine_path = LaunchConfiguration("engine_path")
    camera_device = LaunchConfiguration("camera_device")
    camera_info_url = LaunchConfiguration("camera_info_url")
    frame_id = LaunchConfiguration("frame_id")

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

        # ── camera node ────────────────────────────────────────────────────────
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="camera",
            parameters=[{
                "video_device": camera_device,
                "image_size":   [848, 480],
                "camera_frame_id": frame_id,
                "camera_info_url": camera_info_url,
            }],
        ),

        # ── depth inference ────────────────────────────────────────────────────
        Node(
            package="depth_anything_node",
            executable="depth_anything_node",
            name="depth_anything_node",
            parameters=[{
                "backend":     "tensorrt",
                "engine_path": engine_path,
                "input_size":  364,
            }],
        ),

        # ── visual odometry ────────────────────────────────────────────────────
        # Phase 2: replace this with SPOT body odometry.
        Node(
            package="rtabmap_odom",
            executable="rgbd_odometry",
            name="rgbd_odometry",
            output="screen",
            parameters=[{
                "frame_id":    frame_id,
                "approx_sync": True,
                "queue_size":  5,
            }],
            remappings=[
                ("rgb/image",       "/image_raw"),
                ("rgb/camera_info", "/camera_info"),
                ("depth/image",     "/depth/image"),
                ("depth/camera_info", "/depth/camera_info"),
            ],
        ),

        # ── RGB-D SLAM ─────────────────────────────────────────────────────────
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[{
                "frame_id":        frame_id,
                "subscribe_depth": True,
                "approx_sync":     True,
                "queue_size":      5,
            }],
            remappings=[
                ("rgb/image",         "/image_raw"),
                ("rgb/camera_info",   "/camera_info"),
                ("depth/image",       "/depth/image"),
                ("depth/camera_info", "/depth/camera_info"),
            ],
        ),

        # ── viewer ─────────────────────────────────────────────────────────────
        # Display is in feet (pipeline is metres; rtabmap_viz has no built-in
        # unit switch — apply x3.28084 in the point cloud display settings or
        # use the TF scale trick if needed post-Phase 1b).
        Node(
            package="rtabmap_viz",
            executable="rtabmap_viz",
            name="rtabmap_viz",
            output="screen",
            parameters=[{
                "frame_id":        frame_id,
                "subscribe_depth": True,
                "approx_sync":     True,
                "queue_size":      5,
            }],
            remappings=[
                ("rgb/image",         "/image_raw"),
                ("rgb/camera_info",   "/camera_info"),
                ("depth/image",       "/depth/image"),
                ("depth/camera_info", "/depth/camera_info"),
            ],
        ),
    ])
