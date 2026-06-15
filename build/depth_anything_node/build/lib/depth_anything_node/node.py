import copy
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class DepthAnythingNode(Node):
    def __init__(self):
        super().__init__("depth_anything_node")

        self.declare_parameter("backend", "tensorrt")
        self.declare_parameter("engine_path", "")
        self.declare_parameter("input_size", 364)
        # Only used by the torch backend (laptop / Phase 1a).
        self.declare_parameter("model_name",
                               "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")

        backend    = self.get_parameter("backend").value
        engine_path = self.get_parameter("engine_path").value
        input_size = self.get_parameter("input_size").value
        model_name = self.get_parameter("model_name").value

        self.get_logger().info(f"Loading backend: {backend}")

        if backend == "torch":
            from .backends import TorchBackend
            self._backend = TorchBackend(model_name)
        elif backend == "tensorrt":
            if not engine_path:
                raise ValueError("engine_path must be set when backend=tensorrt")
            from .backends import TensorRTBackend
            self._backend = TensorRTBackend(engine_path, input_size)
        else:
            raise ValueError(f"Unknown backend: '{backend}'. Use 'torch' or 'tensorrt'.")

        self.get_logger().info("Backend ready.")

        self._bridge       = CvBridge()
        self._camera_info  = None

        self._depth_pub = self.create_publisher(Image,      "/depth/image",       1)
        self._info_pub  = self.create_publisher(CameraInfo, "/depth/camera_info", 1)

        self.create_subscription(Image,      "/image_raw",   self._image_cb, 1)
        self.create_subscription(CameraInfo, "/camera_info", self._info_cb,  10)

    def _info_cb(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _image_cb(self, msg: Image) -> None:
        if self._camera_info is None:
            return

        rgb   = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        depth = self._backend.predict(rgb)   # HxW float32, metres

        depth_msg = self._bridge.cv2_to_imgmsg(depth.astype(np.float32), encoding="32FC1")
        depth_msg.header = msg.header

        # Copy so we don't mutate the stored camera_info when patching the header.
        info_msg = copy.copy(self._camera_info)
        info_msg.header = msg.header

        self._depth_pub.publish(depth_msg)
        self._info_pub.publish(info_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthAnythingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
