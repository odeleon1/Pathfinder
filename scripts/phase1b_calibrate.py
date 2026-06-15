#!/usr/bin/env python3
"""
Phase 1b — camera calibration via OpenCV (no ROS needed).

WHY THIS INSTEAD OF ros2 run camera_calibration cameracalibrator
----------------------------------------------------------------
The ROS 2 Jazzy v4l2_camera node does not expose a set_camera_info service,
so the ROS calibration tool crashes immediately. This script talks to the
camera directly via OpenCV, collects checkerboard frames, computes the
calibration, and writes the YAML in the exact format v4l2_camera expects.

USAGE
-----
  python3 scripts/phase1b_calibrate.py

  Controls while running:
    SPACE  — capture the current frame (if checkerboard is detected)
    c      — same as SPACE (capture)
    q      — quit without saving
    Enter  — compute & save calibration (once you have enough frames)

CHECKERBOARD
------------
  Inner corners: 8x6  (a 9x7 grid of squares)
  Square size:   0.025 m (25 mm) — adjust --square if yours differ

  You can print one or display it on a screen.
  Capture at least 15 frames from varied angles and distances.

OUTPUT
------
  ~/.ros/camera_info/<camera_name>.yaml
  Pass to the launch file as:
    camera_info_url:=file:///home/odeleon1/.ros/camera_info/<camera_name>.yaml
"""
import argparse
import sys
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def make_yaml(width, height, K, D, camera_name):
    """Return a ROS 2 camera_info YAML string.

    yaml-cpp (used by ROS 2) requires data arrays as inline sequences:
      data: [v1, v2, ...]
    not block sequences (- v1 / - v2). We build the string manually to
    guarantee the right format instead of relying on PyYAML's dump.
    """
    def fmt(vals):
        return "[" + ", ".join(f"{v:.10g}" for v in vals) + "]"

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k_data = K.flatten().tolist()
    d_data = D.flatten().tolist()[:5]
    r_data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    p_data = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    return f"""\
image_width: {width}
image_height: {height}
camera_name: {camera_name}
camera_matrix:
  rows: 3
  cols: 3
  data: {fmt(k_data)}
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: {fmt(d_data)}
rectification_matrix:
  rows: 3
  cols: 3
  data: {fmt(r_data)}
projection_matrix:
  rows: 3
  cols: 4
  data: {fmt(p_data)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",  type=int,   default=0,     help="V4L2 device index (default: 0)")
    ap.add_argument("--size",    type=str,   default="8x6", help="Inner corners WxH (default: 8x6)")
    ap.add_argument("--square",  type=float, default=0.025, help="Square size in metres (default: 0.025)")
    ap.add_argument("--name",    type=str,   default="camera", help="Camera name (used in YAML filename)")
    ap.add_argument("--min-frames", type=int, default=15,   help="Minimum frames before calibrating")
    args = ap.parse_args()

    cols, rows = [int(x) for x in args.size.split("x")]
    sq = args.square
    min_frames = args.min_frames

    # Prepare object points for one checkerboard view.
    # (0,0,0), (1,0,0), ... scaled by square size, lying in Z=0 plane.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq

    obj_points = []   # 3D points per captured frame
    img_points = []   # 2D points per captured frame

    # Use the default backend (GStreamer on Jetson/JetPack) — do NOT force
    # CAP_V4L2, which bypasses GStreamer and gives black frames on this system.
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open /dev/video{args.device}")

    # Warm up — discard first frames while exposure settles.
    print("Warming up camera ...")
    for _ in range(10):
        cap.read()

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height} from /dev/video{args.device}")
    print(f"Checkerboard: {cols}x{rows} inner corners, {sq*100:.1f} cm squares")
    print(f"Need at least {min_frames} frames. SPACE/c to capture. Enter to calibrate. q to quit.")

    captured   = 0
    last_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        display = frame.copy()
        if found:
            # Refine to sub-pixel accuracy.
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (cols, rows), corners2, found)
            status = f"[DETECTED]  frames: {captured}/{min_frames}  SPACE to capture"
            color = (0, 200, 0)
        else:
            corners2 = None
            status = f"[SEARCHING] frames: {captured}/{min_frames}"
            color = (0, 100, 255)

        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if captured >= min_frames:
            cv2.putText(display, "Press ENTER to calibrate", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Calibration — Pathfinder Phase 1b", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("Quit.")
            break

        elif key in (ord(' '), ord('c')):
            if found and corners2 is not None:
                obj_points.append(objp)
                img_points.append(corners2)
                captured += 1
                print(f"  Captured frame {captured}")
            else:
                print("  No checkerboard detected — move board into view first.")

        elif key == 13:  # Enter
            if captured < min_frames:
                print(f"  Need at least {min_frames} frames (have {captured}). Keep capturing.")
                continue

            print(f"\nCalibrating from {captured} frames ...")
            ret_val, K, D, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, (width, height), None, None
            )

            mean_error = 0.0
            for i in range(len(obj_points)):
                proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, D)
                mean_error += cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
            mean_error /= len(obj_points)

            print(f"  RMS reprojection error: {ret_val:.4f} px")
            print(f"  Mean reprojection error: {mean_error:.4f} px")
            print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

            if ret_val > 1.0:
                print("  WARNING: RMS > 1px — consider recapturing with more varied angles.")

            yaml_str = make_yaml(width, height, K, D, args.name)

            out_dir = Path.home() / ".ros" / "camera_info"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{args.name}.yaml"
            out_path.write_text(yaml_str)

            print(f"\nSaved to: {out_path}")
            print(f"\nUse in launch file:")
            print(f"  camera_info_url:=file://{out_path}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
