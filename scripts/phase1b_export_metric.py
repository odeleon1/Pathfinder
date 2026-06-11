#!/usr/bin/env python3
"""
Phase 1b — export Depth Anything V2-Small METRIC (indoor) to TensorRT FP16.

WHY A SEPARATE SCRIPT FROM PHASE 0
-----------------------------------
Phase 0 exported the *relative* DAv2-Small. The mapper needs *metric* depth —
values in actual metres that are consistent frame-to-frame. The metric model is
the same ViT-S backbone, but trained on indoor scenes (Hypersim) so its output
is already scaled to metres with a sigmoid + max_depth cap. That sigmoid output
is what the ONNX and TRT engine will produce directly, so TensorRTBackend gets
metres with no post-processing needed.

The max_depth cap is 20m — appropriate for indoor scenes (rooms, corridors).

We load from the official HuggingFace transformers repo (safetensors format)
via from_pretrained, then export to ONNX. No raw .pth download needed.

RUN THIS from the project directory with the project .venv:
    ~/Documents/Projects/Pathfinder/.venv/bin/python \
        ~/Documents/Projects/Pathfinder/scripts/phase1b_export_metric.py

PREREQ: authenticated with HuggingFace — run: hf auth login
THEN build the TRT engine (separate step, see pass/fail gate at the end).
"""
import sys
import argparse
from pathlib import Path

HF_MODEL_ID  = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEFAULT_SIZE  = 364   # must be a multiple of 14 (DINOv2 patch size)


def export_onnx(size: int, out: Path) -> None:
    """Load metric DAv2-Small via transformers, export to ONNX."""
    import torch
    from transformers import DepthAnythingForDepthEstimation

    print(f"Loading {HF_MODEL_ID} ...")
    base = DepthAnythingForDepthEstimation.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float32,
    )
    base.eval()

    # Wrap so ONNX export sees a clean tensor-in / tensor-out function.
    # predicted_depth has shape (batch, H, W) — metric depth in metres.
    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, image):
            return self.m(pixel_values=image).predicted_depth

    model = _Wrapper(base)

    # Fixed shape — lets TRT bake dimensions into kernels for max speed.
    dummy = torch.randn(1, 3, size, size, dtype=torch.float32)

    print(f"Exporting to {out} (opset 17, input {size}x{size}) ...")
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["depth"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE,
                    help=f"Input size (multiple of 14). Default: {DEFAULT_SIZE}")
    ap.add_argument("--out", type=str, default=None,
                    help="Output .onnx path. Default: ~/Depth-Anything-V2/")
    args = ap.parse_args()

    if args.size % 14 != 0:
        sys.exit(f"ERROR: --size must be a multiple of 14. Got {args.size}.")

    repo_root = Path.home() / "Depth-Anything-V2"
    onnx_out  = Path(args.out) if args.out else \
        repo_root / f"depth_anything_v2_metric_vits_{args.size}.onnx"

    export_onnx(args.size, onnx_out)

    engine_path = repo_root / f"depth_anything_v2_metric_vits_{args.size}_fp16.plan"

    print("\n=== ONNX export done. Now build the TRT engine ON THIS JETSON: ===")
    print()
    print(f"  trtexec \\")
    print(f"    --onnx={onnx_out} \\")
    print(f"    --saveEngine={engine_path} \\")
    print(f"    --fp16")
    print()
    print("  (No --minShapes/--optShapes/--maxShapes: shapes are baked into the ONNX.")
    print()
    print("trtexec takes 3–10 minutes on the Orin Nano.")
    print()
    print("PASS/FAIL gate after engine build:")
    print(f"  ~/Documents/Projects/Pathfinder/.venv/bin/python \\")
    print(f"    ~/Depth-Anything-V2/benchmark_e2e.py \\")
    print(f"    --engine {engine_path} --size {args.size} --synthetic")
    print()
    print("  PASS: fps > 10 (expect ~40 fps at 364). Values should be 0–20 m.")
    print(f"\nEngine path to use in depth_anything_node launch file:")
    print(f"  engine_path: '{engine_path}'")


if __name__ == "__main__":
    main()
