import numpy as np
import cv2

# ImageNet normalization — DAv2 expects inputs normalized this way.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess(rgb_np: np.ndarray, size: int) -> np.ndarray:
    """HxWx3 uint8 RGB -> 1x3xSxS float32 (ImageNet-normalized, contiguous)."""
    img = cv2.resize(rgb_np, (size, size)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))        # HWC -> CHW
    img = (img - _MEAN) / _STD
    return np.ascontiguousarray(img[None], dtype=np.float32)


class TorchBackend:
    """HuggingFace transformers pipeline — used on the laptop (Phase 1a)."""

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"):
        from transformers import pipeline as hf_pipeline
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = hf_pipeline(task="depth-estimation", model=model_name, device=device)

    def predict(self, rgb_np: np.ndarray) -> np.ndarray:
        from PIL import Image
        result = self._pipe(Image.fromarray(rgb_np))
        return result["predicted_depth"].squeeze().cpu().numpy()  # HxW float32, metres


class TensorRTBackend:
    """
    TensorRT FP16 inference — used on the Jetson (Phase 1b+).

    Loads the metric DAv2-Small engine built by phase1b_export_metric.py.
    predict() takes HxWx3 uint8 RGB and returns HxW float32 depth in metres,
    resized back to the original frame resolution so it's registered to the RGB.
    """

    def __init__(self, engine_path: str, input_size: int = 364):
        import tensorrt as trt
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart  # older cuda-python layout

        self._trt = trt
        self._cudart = cudart
        self._size = input_size

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self._engine = rt.deserialize_cuda_engine(f.read())
        self._ctx = self._engine.create_execution_context()

        # Discover tensor names — don't assume ordering.
        self._in_name = self._out_name = None
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in_name = name
            else:
                self._out_name = name

        self._ctx.set_input_shape(self._in_name, (1, 3, input_size, input_size))
        out_shape = tuple(self._ctx.get_tensor_shape(self._out_name))

        in_nbytes  = int(np.prod((1, 3, input_size, input_size))) * 4   # float32
        out_nbytes = int(np.prod(out_shape)) * 4

        err, self._d_in  = cudart.cudaMalloc(in_nbytes)
        err, self._d_out = cudart.cudaMalloc(out_nbytes)
        self._ctx.set_tensor_address(self._in_name,  int(self._d_in))
        self._ctx.set_tensor_address(self._out_name, int(self._d_out))
        _, self._stream = cudart.cudaStreamCreate()

        self._in_nbytes  = in_nbytes
        self._out_nbytes = out_nbytes
        self._out_shape  = out_shape
        self._out_buf    = np.empty(out_shape, dtype=np.float32)

    def predict(self, rgb_np: np.ndarray) -> np.ndarray:
        orig_h, orig_w = rgb_np.shape[:2]
        x = _preprocess(rgb_np, self._size)

        c = self._cudart
        c.cudaMemcpy(self._d_in, x.ctypes.data, self._in_nbytes,
                     c.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self._ctx.execute_async_v3(self._stream)
        c.cudaStreamSynchronize(self._stream)
        c.cudaMemcpy(self._out_buf.ctypes.data, self._d_out, self._out_nbytes,
                     c.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        depth = self._out_buf.squeeze()   # SxS float32, metres
        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth   # HxW float32, metres
