# Pathfinder — Learning Guide

This document grows alongside the project. Every major concept you encounter here
has a section that explains it from first principles — not just what it is, but why
it exists and how it fits into what we're building.

**How to use it:** read a section when you hit that part of the project. Come back
when something doesn't click. The goal is that nothing stays magic.

---

## Table of Contents

1. [The Core Problem — Why Is This Hard?](#1-the-core-problem)
2. [How Cameras Work](#2-how-cameras-work)
3. [Neural Networks — What "Learning" Actually Means](#3-neural-networks)
4. [Vision Transformers (ViT)](#4-vision-transformers)
5. [Depth Anything V2 — The Full Architecture](#5-depth-anything-v2)
6. [Relative vs Metric Depth](#6-relative-vs-metric-depth)
7. [Inference Optimization — ONNX, TensorRT, FP16](#7-inference-optimization)
8. [ROS 2 — The Robot's Nervous System](#8-ros-2)
9. [SLAM — Building a Map While Getting Lost](#9-slam)
10. [The Jetson Orin Nano — Hardware That Matters](#10-the-jetson-orin-nano)
11. [The Full Pipeline — How It All Connects](#11-the-full-pipeline)

---

## 1. The Core Problem

**What are we trying to do?**

We want SPOT to build a 3D map of its environment as it walks. That requires knowing,
for each camera frame, how far away every visible surface is.

**Why is depth hard to get from a single camera?**

A camera is a machine that collapses 3D reality onto a 2D image. Every point in the
scene projects onto a pixel, and in that projection, the distance information is
*destroyed*. A tiny nearby object and a huge distant object can land on the exact
same pixel — the image alone cannot tell them apart.

Think of looking at a full moon and a tennis ball held at arm's length. They subtend
roughly the same angle (about 0.5°). In a photo with no other context, they look the
same size. You know the moon is bigger because your brain applies world knowledge,
not because the image told you.

That is exactly what a depth network does — it applies learned world knowledge.

**Why not just use a depth camera?**

We could. A RealSense or similar uses structured light or time-of-flight to measure
depth directly. It's simpler and more accurate. But this project deliberately uses
monocular depth estimation because:

1. The Arducam IMX462 is a tiny, lightweight night-vision camera that SPOT can carry
   easily. Depth cameras are heavier and power-hungry.
2. Learning monocular depth is the intellectually interesting hard problem.
3. The depth network gives us per-pixel inference — it reasons about the whole scene
   simultaneously, not just what a laser beam happens to hit.

The tradeoff: estimated depth is noisier and can be wrong in ways a real sensor
never would be. The map will look rougher. That's a known cost.

---

## 2. How Cameras Work

### The pinhole model

Every camera — from your phone to a $50,000 cinema lens — is modeled at its core
as a **pinhole camera**. Imagine a light-tight box with a tiny hole on one side.
Light rays from the scene pass through the hole and project onto the opposite wall,
forming an upside-down image.

Real cameras replace the pinhole with a lens (to gather more light) and the back wall
with a sensor, but the geometry is the same.

### Focal length

The focal length `f` determines how much the scene is magnified. A long focal length
makes objects appear larger and compresses apparent depth (telephoto look). A short
focal length gives a wide field of view.

In pixel units: a larger `f` means a point in 3D space projects to a pixel that is
farther from the center of the image.

### The intrinsic matrix K

The **camera intrinsic matrix** (called K) captures everything about a specific
camera's optics. It's a 3×3 matrix:

```
K = [ f_x   0   c_x ]
    [  0   f_y  c_y ]
    [  0    0    1  ]
```

- `f_x`, `f_y`: focal length in pixels (x and y, usually equal for square pixels)
- `c_x`, `c_y`: principal point — the pixel where the optical axis hits the sensor
  (ideally the image center, but not always)

Given a 3D point (X, Y, Z) in the camera's coordinate frame, its pixel location is:

```
u = f_x * (X/Z) + c_x
v = f_y * (Y/Z) + c_y
```

Notice Z divides out. That's the depth information being lost.

**Camera calibration** (Step 4 of Phase 1b) is the process of finding K for your
specific camera by photographing a checkerboard at known positions. Without K,
you can't convert the 2D pixel coordinates from the depth map back into metric 3D
coordinates — the whole reconstruction depends on it.

### Distortion

Real lenses aren't perfect pinholes. They introduce **distortion**: straight lines
in the world appear curved in the image (barrel distortion on wide-angle lenses,
pincushion on telephoto). Calibration also estimates distortion coefficients to
correct for this.

### Rolling shutter

Most cheap cameras (including the Arducam IMX462) use a **rolling shutter**:
instead of capturing the entire frame at one instant, they scan the sensor from
top to bottom one row at a time. This takes a few milliseconds.

If the camera is moving (like SPOT walking), the top and bottom of the image were
captured at slightly different camera positions. This warps the image — straight
lines appear tilted or curved during fast motion. It's a known source of error in
the reconstruction and something Phase 4 will address.

---

## 3. Neural Networks

### What a neural network is

A neural network is a function that maps inputs to outputs. That's it. The "neural"
part is just a specific way of building that function — stacked layers of linear
transformations followed by non-linear activations.

One layer computes:
```
output = activation(W · input + b)
```
where W is a matrix of **weights** and b is a **bias** vector.

Stack dozens of these layers and the composed function can approximate almost
anything — including "given this RGB image, what is the depth of every pixel."

The weights W and biases b are the **parameters** — millions of numbers that define
what the function does. Before training, they're random noise. The function outputs
garbage. Training is the process of finding good values for them.

### How training works (gradient descent)

You need:
1. A **dataset**: lots of examples of (input, correct output) pairs. For depth:
   millions of (RGB image, depth map) pairs from LiDAR sensors or synthetic renders.
2. A **loss function**: a way to measure how wrong the current output is.
   Example: mean squared error between predicted depth and true depth.
3. An **optimizer**: a rule for adjusting weights to reduce the loss.

The standard optimizer is **gradient descent**. The gradient of the loss with respect
to every weight tells you which direction to move each weight to make the loss smaller.
You take a small step in that direction. Repeat millions of times. The function gets
better at predicting depth.

**Backpropagation** is the algorithm that computes those gradients efficiently.
It uses the chain rule from calculus to propagate the error signal backward through
every layer, computing how much each weight contributed to the error.

You never need to implement backprop — PyTorch does it automatically. But knowing it
exists explains why training needs a GPU (billions of multiply-accumulate operations),
why training takes days or weeks, and why we don't train from scratch.

### Training vs inference

**Training**: forward pass (input → output) + backward pass (compute gradients) +
weight update. Expensive. Needs labeled data. Done once (by the research team).

**Inference**: forward pass only. No gradients, no weight updates. The weights are
fixed. This is what Pathfinder does at runtime — read weights from disk, run image
through network, get depth map.

Inference is much cheaper than training, but still needs a GPU for speed.

### Transfer learning

Training Depth Anything V2 from scratch would require:
- Tens of millions of images with depth labels
- Weeks on hundreds of GPUs
- Millions of dollars in compute

We don't do that. Instead, we use **transfer learning**:

1. Someone else (the DINOv2 team at Meta) pre-trained a ViT backbone on 142M images
   using a self-supervised method (no labels needed). The backbone learned to extract
   rich visual features — edges, textures, objects, spatial relationships.

2. The DAv2 team took that backbone, attached a depth-prediction head, and fine-tuned
   the whole thing on depth datasets. The backbone already understood images; it just
   needed to learn to output depth.

3. For the metric variant, they fine-tuned again on Hypersim (a synthetic indoor
   dataset with precise metric depth labels).

Each step is far cheaper than training from scratch because you start from a good
initialization. This is why a 25M parameter model trained by a team of researchers
outperforms what you could train yourself in a weekend.

---

## 4. Vision Transformers

### The Transformer — originally for text

Transformers were invented for natural language processing (the "T" in GPT, BERT, etc.).
The core insight: to understand a word, you need context from the whole sentence.
"Bank" means something different in "river bank" vs "bank account." A model needs to
look at *all other words* to figure out which meaning applies.

The mechanism for doing this is called **self-attention**.

### Self-attention — the key idea

Given a sequence of tokens (words, or image patches), self-attention lets each token
look at every other token and decide how much to borrow from it.

Each token produces three vectors:
- **Query (Q)**: "what information am I looking for?"
- **Key (K)**: "what information do I contain?"
- **Value (V)**: "what should I share if someone attends to me?"

The attention score between token i and token j is how well i's query matches j's key:
```
score(i, j) = dot(Q_i, K_j) / sqrt(dimension)
```

Apply softmax to all scores for token i → you get attention weights (sum to 1).
Token i's new representation = weighted sum of all tokens' V vectors.

The division by `sqrt(dimension)` keeps the dot products from getting too large,
which would cause softmax to saturate (output near 0 or 1 for everything, killing
the gradient signal during training).

**Why this matters**: every token gets global context in *every layer*, not just
after many layers of local operations like CNNs. A dark pixel near a window can
attend to the bright outdoor scene and infer it's far away — in layer 1.

### From text to images — Vision Transformer (ViT)

The transformer was adapted for images by treating **patches** as tokens.

1. **Split**: divide the image into a grid of non-overlapping patches (14×14 pixels
   for DAv2, chosen because the ViT was pretrained with DINOv2 which uses that size).

   At 364×364 input: 364 / 14 = 26 patches per side → 26 × 26 = **676 patches total**.
   
   This is why **input size must be a multiple of 14** — any remainder means an
   incomplete patch, which breaks the patch embedding.

2. **Embed**: each 14×14×3 patch (588 numbers) is linearly projected to a 384-dimensional
   vector. This is just a matrix multiply — learned during training. Now you have
   676 vectors of size 384.

3. **Position embeddings**: transformers have no inherent sense of order (unlike
   RNNs). You add a learned position embedding to each token so the model knows
   where in the image each patch came from. Without this, a patch from the top-left
   corner is identical to one from the bottom-right.

4. **Transformer layers**: 12 layers of self-attention + feedforward networks.
   Each layer refines the token representations with global context.

5. **Output**: 676 tokens, each now containing rich contextual information about
   its patch and how it relates to the rest of the image.

### ViT-S (Small) — what "Small" means

The ViT comes in sizes that differ in embedding dimension and number of layers:

| Variant | Embedding dim | Layers | Heads | Parameters |
|---------|--------------|--------|-------|------------|
| ViT-S   | 384          | 12     | 6     | ~21.5M     |
| ViT-B   | 768          | 12     | 12    | ~86M       |
| ViT-L   | 1024         | 24     | 16    | ~307M      |

ViT-S (Small) is the only one that fits in the Jetson's 8GB shared RAM with the
rest of the pipeline running. This is a hard constraint, not a performance choice.

### Multi-head attention

In practice, each attention layer runs **multiple attention heads** in parallel
(6 heads in ViT-S). Each head learns to attend to different kinds of relationships
— one might specialize in edges, another in object boundaries, another in depth cues.
Their outputs are concatenated and projected back to the embedding dimension.

---

## 5. Depth Anything V2

### Overview

DAv2 has two parts: a **backbone** that understands the image, and a **head** that
converts that understanding into a depth map.

```
RGB image (H × W × 3)
        │
  [ Preprocessing ]
  resize → 364×364
  ImageNet normalize
        │
  [ ViT-S Backbone ]         ← 12 layers of self-attention
  676 tokens, dim=384
        │
  [ DPT Head ]               ← multi-scale fusion + upsample
        │
  Depth map (364 × 364)      ← float32, metres (metric model)
```

### The DPT Head — from tokens to pixels

After the backbone, you have 676 tokens of size 384. That's not a depth map —
it's an abstract sequence. The **DPT (Dense Prediction Transformer) head** converts
it back to a spatial output.

The key insight of DPT: don't just use the *last* transformer layer's output.
Use intermediate layers too, because different layers capture different things:
- **Early layers** (layer 3): low-level details — edges, textures, fine structure
- **Middle layers** (layer 6, 9): mid-level patterns — object parts, surfaces
- **Final layer** (layer 12): high-level semantics — what kind of object, spatial context

DPT hooks into 4 layers and extracts their token representations. Each set of
tokens is reshaped back into a 2D spatial grid (26×26 for 364-input) and then
progressively upsampled through convolutional blocks, combining features at each
scale. The final output is upsampled to the full 364×364 resolution.

This multi-scale fusion is what makes DPT good at dense prediction. Using only
the last layer would lose fine-grained edge information. Using all layers gives
you both the "where is the depth change" detail and the "what kind of surface is
this" context.

### ImageNet normalization — why those specific numbers

Before the image enters the network, you normalize each channel:

```python
img = img / 255.0                          # [0, 255] → [0.0, 1.0]
img = (img - [0.485, 0.456, 0.406]) \      # subtract per-channel mean
       / [0.229, 0.224, 0.225]             # divide by per-channel std
```

The numbers `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]` are the mean
and standard deviation of the ImageNet dataset, computed across millions of images,
per color channel (R, G, B).

**Why use ImageNet stats?** The ViT-S backbone was pretrained on ImageNet by DINOv2.
All of its internal weight values were learned assuming this normalization applied.
The weights encode patterns like "a normalized pixel value of +1.5 in the red channel
combined with −0.3 in the blue channel looks like a sky region." If you feed the
network differently normalized pixels, those patterns no longer match and the
network's features degrade.

Think of it like a recipe calibrated for Celsius. If you use Fahrenheit by mistake,
all the temperature steps are wrong and the dish fails.

---

## 6. Relative vs Metric Depth

### Relative depth — proportional but unitless

The base DAv2 model outputs **relative depth**. Each pixel gets a value that is
*proportional* to depth, but there are no real-world units attached.

Problems:
1. **No scale**: if pixel A has value 0.8 and pixel B has 0.4, B is about twice as
   far — but is that 0.5m vs 1m, or 5m vs 10m? You don't know.
2. **Per-frame scale drift**: the scale can change between frames. Frame 1 might
   output values in the range [0.1, 0.9] meaning "1m to 9m," and frame 2 might
   output [0.1, 0.9] meaning "0.5m to 4.5m." The mapper sees inconsistent geometry.

Relative depth is fine for single-image tasks like "detect what's closer." It's
useless for building a consistent multi-frame map.

### Metric depth — actual metres

The metric variant adds a final **sigmoid + scale** operation:

```python
# The network outputs a raw value for each pixel (any real number)
# sigmoid maps it to (0, 1):
normalized = sigmoid(raw_output)   # = 1 / (1 + exp(-raw_output))
# scale to metres:
depth_metres = max_depth * normalized   # max_depth = 20.0 for indoor
```

This bounds the output to (0, 20) metres. The network was fine-tuned on Hypersim
(a photorealistic synthetic indoor dataset from Adobe Research) where real metric
depth labels were available. Fine-tuning teaches the network to produce values in
actual metres that are consistent across frames.

**Hypersim** is synthetic — it's computer-generated photorealistic indoor scenes
(apartments, offices, corridors) where every pixel's exact depth is known because
the 3D scene is defined analytically. This gives perfect ground-truth labels that
you can't get reliably from real sensor noise. The tradeoff is a slight sim-to-real
gap: the network was trained on synthetic images, so it may not generalize perfectly
to all real-world lighting and materials. In practice it works well for indoor scenes.

**max_depth = 20m** is appropriate for indoor spaces (rooms and corridors don't
exceed 20m). If this were an outdoor project, you'd use the VKITTI-trained variant
with a higher cap (80m for driving scenes).

---

## 7. Inference Optimization

### Why not just run PyTorch on the Jetson?

You could load the PyTorch model directly and run it. It would work. But PyTorch
executes operations through a general-purpose CUDA runtime: it launches kernels
one by one, allocates memory on the fly, and doesn't know what's coming next.

**TensorRT is a compiler, not a runtime.** It analyzes the entire computation graph
in advance, knows the exact input size (because we fixed it at 364×364), and can:

1. **Fuse layers**: a convolution followed by batch normalization followed by ReLU
   is three separate operations in PyTorch. TRT fuses them into one CUDA kernel —
   one GPU launch, one memory pass. Reduces latency by eliminating kernel launch
   overhead and intermediate memory writes.

2. **Select optimal kernels**: for each operation, TRT benchmarks multiple CUDA
   kernel implementations (different tile sizes, memory layouts, etc.) and picks
   the fastest one for *your specific GPU*. This is why building an engine takes
   5–10 minutes — it's running benchmarks, not just compiling.

3. **Plan memory**: TRT knows every tensor's lifetime in advance and schedules them
   to minimize GPU memory usage (reusing buffers when their lifetimes don't overlap).

The result: on the Jetson Orin Nano, PyTorch might run the model at ~15–20 fps.
TRT runs it at ~70 fps. Same weights, same math, radically different execution.

### ONNX — the bridge format

You can't hand PyTorch code directly to TRT. You need to express the model as a
graph of standard operations that TRT understands. **ONNX** (Open Neural Network
Exchange) is that standard format.

`torch.onnx.export` **traces** the model: it runs a dummy input through and records
every mathematical operation — matrix multiplies, activations, reshapes, normalizations.
The resulting graph is serialized to an `.onnx` file.

When the model is large (like a ViT-S), PyTorch stores the weights in a separate
`.data` file next to the `.onnx`. The `.onnx` has the graph structure; the `.data`
has the numbers. TRT reads both.

The onnxscript warnings during export ("No Adapter To Version $17 for Resize") are
a non-fatal version conversion issue — the model ends up in opset 18, which TRT 10.x
handles fine.

### FP16 — half precision

A **float32** number uses 32 bits (4 bytes) to represent a decimal value.
A **float16** number uses 16 bits (2 bytes) — half the range, half the precision.

For deep learning inference, FP16 is almost always accurate enough. The model was
trained with FP32 precision but the output differences at FP16 are below the noise
floor of depth estimation.

The gains are real:
- **2× memory**: every weight, every activation tensor is half the size. On a device
  with 8GB shared RAM (both CPU and GPU use the same physical memory), this matters.
- **2× throughput**: the Orin Nano's GPU has **Tensor Cores** — special hardware
  units that execute matrix multiplications on FP16 data in bulk. Two FP16 ops per
  clock cycle instead of one FP32 op. The 70 fps result is Tensor Cores at work.

`--fp16` in trtexec tells TRT it's allowed to run layers in FP16. TRT decides
per-layer whether FP16 is safe (it is for all ViT operations).

### Fixed vs dynamic shapes

When exporting ONNX, we did not specify `dynamic_axes`. This means the input shape
`(1, 3, 364, 364)` is **baked into the ONNX graph**.

TRT can further optimize a fixed-shape engine: it knows at compile time exactly how
many elements every tensor has, which enables tighter kernel scheduling and eliminates
all shape-checking logic at runtime.

The consequence: you cannot feed this engine a different-sized image. It will always
expect exactly 364×364. Our `TensorRTBackend` resizes every input to 364×364 before
inference and resizes the depth output back to the original camera resolution afterward.

---

## 8. ROS 2

### What ROS 2 is (and isn't)

**ROS** stands for Robot Operating System, but it's not an OS. It's a **middleware**
framework for building robot software. It provides:

- A standardized way for software components to communicate
- A library of common robot data types (images, point clouds, transforms, odometry)
- Tools for logging, visualization, parameter management, and launching systems

The "2" matters: ROS 2 replaced ROS 1 with a proper real-time-capable foundation
(DDS — Data Distribution Service), better security, and support for modern Ubuntu.
We use **Jazzy**, the 2024 LTS release.

### Nodes and topics — the pub/sub pattern

A ROS 2 **node** is a process that does one job. The Pathfinder pipeline has five:

- `v4l2_camera_node` — reads frames from the USB camera
- `depth_anything_node` — runs the depth network
- `rgbd_odometry` — estimates camera pose from RGB+depth
- `rtabmap` — builds and maintains the 3D map
- `rtabmap_viz` — displays the map

Nodes communicate through **topics** — named channels. One node **publishes** to
a topic; any number of nodes can **subscribe** to receive those messages.

This is a **publish/subscribe pattern**: the publisher doesn't know or care who is
listening. The subscriber doesn't know or care who is sending. They're decoupled.

```
v4l2_camera_node  ──publishes──▶  /image_raw        ──▶  depth_anything_node
                  ──publishes──▶  /camera_info       ──▶  depth_anything_node
                                                     ──▶  rgbd_odometry

depth_anything_node ──publishes──▶  /depth/image     ──▶  rgbd_odometry
                    ──publishes──▶  /depth/camera_info ──▶ rgbd_odometry

rgbd_odometry ──publishes──▶  /odom               ──▶  rtabmap
                              /tf (transforms)

rtabmap ──publishes──▶  /rtabmap/cloud_map      ──▶  rtabmap_viz
```

### Message types

Every topic carries a specific **message type**. Some relevant ones:

- `sensor_msgs/Image`: an image frame. Contains width, height, encoding (e.g. `bgr8`
  for 3-channel 8-bit, `32FC1` for 1-channel 32-bit float), and the raw pixel data.
  We publish depth as `32FC1` — one 32-bit float per pixel, in metres.

- `sensor_msgs/CameraInfo`: the camera's K matrix, distortion coefficients, and
  resolution. Published alongside every image so downstream nodes can do 3D geometry.

- `nav_msgs/Odometry`: the camera's pose (position + orientation) and velocity.

The **header** on every message contains a **timestamp** and **frame_id** (which
coordinate frame the data lives in). Getting timestamps right is critical because
rtabmap's `approx_sync` matches RGB and depth frames by timestamp — if they don't
match within a tolerance, the frame pair is dropped silently.

### Coordinate frames and TF

In a robot system, you have many coordinate frames: camera frame, robot body frame,
world frame, etc. The **TF system** maintains the transformations between all of them.

For example, to project a depth pixel into world coordinates, you need to know:
- pixel → camera frame (from K matrix)
- camera frame → robot body (from mount geometry)
- robot body → world (from odometry)

Each node publishes its piece of this chain. The TF library lets any node look up
any combination at any timestamp.

**The frame_id chain — how naming must be consistent**

Every ROS 2 sensor message has a `header.frame_id` field: a string naming which
coordinate frame the data is expressed in. For the camera pipeline:

- `/image_raw` and `/camera_info` both have `header.frame_id = <camera frame name>`
- `/depth/image` and `/depth/camera_info` inherit the same name

rtabmap_ros's `rgbd_odometry` node has a `frame_id` parameter that means the
**robot's base frame** — the body frame that moves through the world. It looks up TF:

```
TF lookup: base_frame → camera_frame
```

This tells it where the camera is mounted relative to the body. For a handheld
device where the camera *is* the robot, these should be the same frame, and the
lookup is identity.

**The gotcha that bit Phase 1b:**

Our launch file defaulted `frame_id` to `camera_optical_frame`. But v4l2_camera
publishes camera_info with the `frame_id` taken from the **`camera_name` field in
the calibration YAML** — not from the `camera_frame_id` ROS parameter as you might
expect. Our YAML had `camera_name: camera`. So:

- rtabmap thought the base frame was `camera_optical_frame`
- camera_info said the camera frame was `camera`
- rtabmap tried: `lookupTransform("camera_optical_frame", "camera")` → failed
- Error: `getting transform camera_optical_frame -> camera: target_frame does not exist`

Fix: set `frame_id: camera` in the launch (to match YAML camera_name), so base
frame == camera frame == `camera`. Identity lookup, no TF publisher needed.

The broader principle: when this error appears, the fix is always to trace the
`frame_id` from the YAML → camera_info header → rtabmap `frame_id` param and
make sure they all agree.

### Launch files

Running 5 nodes manually with the right parameters each time is error-prone.
**Launch files** are Python scripts that define the full system configuration:
which nodes, what parameters, what remappings (connecting a topic under one name
to a subscriber expecting a different name).

`pipeline.launch.py` is Pathfinder's launch file. It declares arguments (engine
path, camera device, calibration URL) with defaults, then instantiates all 5 nodes.
Running `ros2 launch depth_anything_node pipeline.launch.py` starts everything.

### approx_sync — why it matters

`rgbd_odometry` and `rtabmap` need a synchronized (RGB image, depth map) pair —
frames captured at the same time. But the camera publishes images and our node
publishes depth on separate topics, at slightly different times.

`approx_sync: True` tells rtabmap to match the nearest-in-time RGB and depth frames
within a tolerance window, rather than requiring exact timestamp equality. Without
this, no frames would ever match (exact equality is impossible with async topics).

The cost: very small temporal misalignment between the RGB and depth. In practice
invisible.

### What a topic actually costs

The pub/sub picture above ("node A publishes, node B subscribes") hides something that
turns out to dominate this project's performance: **a topic is not a shared variable.**

When node A publishes an image to node B in a different process, the message is
*serialized* — flattened into a byte buffer — handed to the operating system, copied
through the loopback network stack, then *deserialized* back into an object on the
other side. For a small message (a pose, a number) that cost is irrelevant. For an
848×480 RGB image it is 1.22 MB of copying, per subscriber, per frame.

That last part is the trap. Costs scale with the number of *subscribers*, not topics.
Our original Phase 1b graph had five separate processes, and both image topics had
three or four subscribers each:

```
/image_raw   (1.22 MB) → depth node, odometry, rtabmap, viewer   = 4.9 MB
/depth/image (1.63 MB) → odometry, rtabmap, viewer               = 4.9 MB
                                                        ~9.8 MB per frame
```

At 10 Hz that is ~98 MB/s of pure serialize-copy-deserialize work, none of which is
doing anything useful. This is why the pipeline ran at 7–10 Hz even though the camera
could do 27 Hz and the depth network could do 32 fps. **The bottleneck was not compute.
It was talking.**

The lesson generalizes: when a ROS 2 pipeline is slower than every one of its parts
measured individually, suspect the plumbing before you optimize the algorithms.

### Composition and intra-process comms

ROS 2's answer is **composition**: instead of running each node as its own process, you
load several nodes as plugins into one shared process called a *container*.

```
  Separate processes                One container
  ┌────┐  serialize   ┌────┐        ┌──────────────────────┐
  │ A  │ ───────────► │ B  │        │  A ──(pointer)──► B  │
  └────┘   1.22 MB    └────┘        └──────────────────────┘
```

With `use_intra_process_comms: true`, a publish to a subscriber in the same process
becomes **pointer passing**. The message object is never flattened, never copied, never
touches the network stack. The subscriber gets a reference to the exact same bytes in
memory. Cost goes from ~1.22 MB of copying to ~8 bytes.

For this to be legal, a node has to be written as a *component* — a class that can be
loaded dynamically, rather than a program with its own `main()`. Most standard ROS 2
nodes ship both ways. Our own `depth_anything_node` cannot join, because it is written
in Python and `rclpy` has no zero-copy intra-process support. That's fine — it means
exactly two messages per frame still cross a process boundary (RGB out to the depth
node, depth back), instead of seven.

Two hard-won practicalities:

- **Use `component_container_mt`, not `component_container`.** The single-threaded
  container runs everything on one executor thread. If one component occupies that
  thread (a camera capture loop, say), the container can no longer answer the service
  call that loads the *next* component — and startup hangs forever. The `_mt` variant
  runs a multi-threaded executor and doesn't have this failure mode.
- **Composition is not free of new failure modes.** We could not put the camera node in
  the container at all: `image_transport` loads its plugins lazily via `dlopen` when
  subscribers appear, which races the container's own `dlopen` for the next component
  and deadlocks the process permanently. Two libraries loading concurrently is a classic
  dynamic-linker hazard, and it produces a hang with no error message.

### DDS, and why big messages vanish

Under ROS 2's API sits **DDS** — the actual middleware that moves bytes. ROS 2 doesn't
implement networking itself; it defines an abstraction (`rmw`) and ships a DDS
implementation underneath, FastDDS by default. When two processes exchange a message,
DDS is what carries it, typically over UDP on the loopback interface even when both
processes are on the same machine.

UDP sockets have a fixed-size kernel buffer. If a 2 MB message arrives and the buffer
holds 208 KB, the excess is **silently discarded** — UDP has no retransmission and no
error. The symptom is not a crash or an error log. The symptom is a subscriber that
just... never receives anything, while the publisher reports everything as fine.

Ubuntu's defaults are sized for ordinary network traffic, not for robots shipping
uncompressed images:

```bash
net.core.rmem_default = 212992   # 208 KB — receive, the value DDS actually uses
net.core.rmem_max     = 212992   # 208 KB — receive, the ceiling a socket may request
net.core.wmem_default = 212992   # 208 KB — send
net.core.wmem_max     = 212992   # 208 KB — send
```

**All four matter, and the `_max` ones are the least important.** This is the part that
cost us an hour. `rmem_max` is only a *ceiling* — it permits a socket to ask for a
bigger buffer, it doesn't give it one. FastDDS sizes its sockets from `rmem_default`.
So raising `rmem_max` alone changes nothing whatsoever, while looking exactly like the
right fix. And raising both receive values still isn't enough, because the *sending*
side has its own buffer (`wmem_*`) and a publisher that can't push a 2 MB message out
fails just as completely as a subscriber that can't take it in.

```bash
sudo tee /etc/sysctl.d/60-ros2.conf >/dev/null <<'EOF'
net.core.rmem_max=16777216
net.core.rmem_default=16777216
net.core.wmem_max=16777216
net.core.wmem_default=16777216
EOF
sudo sysctl --system
```

There's a nastier lesson buried here. This bug was *latent* for all of Phase 1b — the
viewer worked fine. It only appeared once the pipeline got faster, because at 7–10 Hz
the traffic fit and at 20 Hz it didn't. **Making a system
faster can expose bugs that were always there.** When something breaks right after an
optimization, "the optimization broke it" and "the optimization revealed it" are very
different diagnoses, and they lead to opposite fixes.

### Fan-out — the part that isn't about size

Raising the buffers was necessary but not sufficient, and the reason is worth
internalizing: **the limit is how many subscribers a topic has, not how big the message
is.**

A large topic serves its *first* subscriber at full rate and starves the rest. Measured
on `/image_raw` (1.22 MB/frame) with two subscribers already attached, a third got
0.20 Hz. The identical data on a topic with one subscriber ran at 24 Hz. Nothing about
the bytes changed — only how many readers were competing for them.

Two fixes follow, attacking different halves of the problem:

```
compress it                        give it its own topic
1220 KB -> 107 KB                  /image_raw -> /viz/image_raw
0.20 Hz -> 24.40 Hz                0.20 Hz -> 24.13 Hz
```

`image_transport` does both without touching your nodes. Publishers automatically offer
`<topic>/compressed` once the plugin is installed, and `image_transport republish` will
decode a compressed stream onto a fresh topic for a single consumer.

The general principle: **when a subscriber starves, count the other subscribers before
you look at the message size.** Size determines when the problem shows up; fan-out
determines who it hits.

### Not everything escapes a container

Composition has a cost the tutorials don't mention: topics published from inside an
intra-process container may not reach subscribers outside it. Measured on this pipeline:

```
/tf     17.7 Hz   fine
/odom    0.13 Hz  starved   (a few hundred bytes!)
```

`/odom` is tiny, so this is not the fan-out effect above — it is the intra-process
publisher itself. The practical consequence: **decide early which topics need to leave
the container**, because composition is not transparent to outside consumers. Here it
costs nothing (nothing outside needs `/odom`), but it is exactly the kind of thing that
bites when a new consumer shows up later.

### Message encodings — depth as 16UC1 vs 32FC1

Depth images have two conventional wire formats in ROS:

| Encoding | Meaning | Bytes/pixel | Range at 848×480 |
|---|---|---|---|
| `32FC1` | 32-bit float, **metres** | 4 | 1.63 MB/frame |
| `16UC1` | 16-bit unsigned int, **millimetres** | 2 | 815 KB/frame |

`32FC1` is the obvious choice — it's the native output of the network and needs no
conversion. But on a bandwidth-bound pipeline, halving a topic's size is a bigger win
than saving a millisecond of conversion, so we publish `16UC1`.

The precision question is the one worth thinking through: 16-bit millimetres quantizes
depth to 1 mm steps and caps at 65.535 m. Is that lossy? Technically yes. Practically
irrelevant — a monocular depth *network's* error at 5 m is on the order of tens of
centimetres. Quantizing to 1 mm is like rounding a rough estimate to more decimal places
than the estimate deserves. The rule: **match your precision to your actual error, not
to your data type.**

One real subtlety. Converting metres→millimetres means handling NaN, negatives, and
overflow. The obvious numpy spelling does four passes over 400k pixels:

```python
mm = np.nan_to_num(depth, nan=0.0) * 1000.0
out = np.clip(mm, 0, 65535).astype(np.uint16)     # 2.9 ms
```

OpenCV does it in one SIMD, multithreaded pass:

```python
out = cv2.multiply(depth, 1000.0, dtype=cv2.CV_16U)   # 0.85 ms
```

And its saturation behaviour is *better*: NaN, negative and overflowing values all land
on 0, which every ROS depth consumer already reads as "no measurement". The numpy
version clips overflow to 65535 — a confident-looking 65 m reading that rtabmap would
happily insert into the map. **A wrong value is worse than a missing one**, because the
missing one is handled by existing code paths and the wrong one isn't.

---

## 9. SLAM

### The problem: you don't know where you are

A map is a spatial record of where things are. To make a map, you need to know where
*you* are when you observe each thing. But knowing where you are requires a map.
This circular dependency is the **SLAM problem**: Simultaneous Localization And Mapping.

SLAM algorithms solve both at once, updating position estimates and the map together
as new observations arrive. It's iterative — early estimates are uncertain, and the
map and pose estimates get progressively more consistent.

### Visual odometry

**Odometry** is the estimation of position change from sequential measurements.
**Visual odometry** does this from camera images.

`rgbd_odometry` in rtabmap takes consecutive (RGB, depth) frame pairs and estimates
how the camera moved between them by:

1. Detecting **feature points** in the current frame (corners, blobs, distinctive
   texture regions using ORB or SIFT-style detectors).
2. **Matching** those features to features in the previous frame (by descriptor
   similarity).
3. **Solving for the transform** (rotation + translation) that minimizes reprojection
   error: given matched feature positions in 2D and their 3D positions (from depth),
   what camera motion explains the observation?

The result is a pose estimate: camera position + orientation relative to the previous
frame. Integrating these incremental estimates gives a trajectory.

**Drift** is the fatal flaw of pure odometry. Small errors in each frame estimate
accumulate. After walking 100 meters in a loop and returning to the start, the
estimated position might be 2 meters off. The map "drifts" — it's internally
consistent per-frame but wrong globally.

### Loop closure — the key insight

When the camera revisits a place it has been before, a **loop closure** detector
recognizes it (by comparing the current frame's visual features to a database of
previously seen frames). This recognition gives a *constraint*: "the current pose
must be near the pose when I first saw this place."

Applying this constraint propagates a correction backward through the entire pose
history, reducing the accumulated drift. After loop closure, the map snaps into
global consistency.

Loop closure is why SLAM works at scale. Without it, all you have is drifting
odometry. With it, you get a map that can be walked multiple times and stays coherent.

### Dense vs sparse maps

- **Sparse map**: only the tracked feature points — a point cloud with a few thousand
  points. Useful for odometry and loop closure, but not for obstacle avoidance or
  inspection — you can't see surfaces.

- **Dense map**: depth from every pixel of every frame, projected into 3D. Millions
  of points. You can see the actual geometry of walls, floors, and objects.

Pathfinder builds a **dense map** by publishing full depth frames. rtabmap accumulates
these, colors each 3D point with the corresponding RGB pixel, and outputs a colored
point cloud. That's what you see in rtabmap_viz.

### RTAB-Map

**RTAB-Map** (Real-Time Appearance-Based Mapping) is a full SLAM system designed
specifically for online, memory-managed mapping. Its key feature: it manages a
working memory of recently seen places and long-term memory of potentially loop-closable
frames, pruning stale data to maintain real-time performance even in large environments.

For Pathfinder, rtabmap is doing:
1. RGB-D odometry (frame-to-frame pose)
2. Global map optimization (poses + 3D point cloud)
3. Loop closure detection
4. Point cloud assembly and colorization

In Phase 2, we'll replace rtabmap's visual odometry with SPOT's body odometry,
which is more accurate and immune to poor visual texture (dark corridors, featureless
walls).

---

## 10. The Jetson Orin Nano

### Why the hardware matters

Most neural network tutorials run on a desktop GPU with 8–16GB of *dedicated* VRAM
and plenty of CPU RAM separately. The Jetson is fundamentally different.

### Shared memory architecture

The Jetson Orin Nano has **8GB of LPDDR5 RAM shared between the CPU and GPU**. There
is no separate VRAM. When the GPU runs inference, it uses the same physical memory
as the operating system, ROS 2, and everything else.

This has big implications:
- **8GB total** for OS (~1GB) + ROS 2 nodes (~2GB) + depth model weights (~95MB) +
  TRT activations (~200MB) + rtabmap map data + everything else.
- The depth model must be the **Small variant** — Base or Large don't leave enough
  room for the rest of the pipeline.
- FP16 (half precision) is mandatory partly because it halves the GPU memory footprint.

### CUDA and Tensor Cores

CUDA is NVIDIA's parallel computing platform. The Jetson's GPU has **CUDA cores** for
general floating-point computation and **Tensor Cores** for matrix multiplications
specifically.

Transformer self-attention is mostly large matrix multiplies. Tensor Cores are
purpose-built for exactly this operation:
- Input: two FP16 matrices
- Output: FP32 accumulation (to avoid precision loss in the sum)
- Throughput: far higher than running FP32 on regular CUDA cores

This is the hardware reason FP16 isn't just "smaller" — it unlocks a fundamentally
faster execution path. The 70 fps result would drop to ~35 fps in FP32 on the same
hardware.

### Compute Capability 8.7

TRT reported "Compute Capability: 8.7" for the Orin Nano's GPU. This is NVIDIA's
versioning for GPU instruction sets. CC 8.7 is Ampere-generation, which supports:
- FP16 Tensor Cores (needed for our inference path)
- Structured sparsity (optional future optimization)
- INT8 quantization (future fallback if FP16 isn't fast enough)

TRT engines are compiled for a specific CC. A `.plan` file built on the Orin (CC 8.7)
will not run on a different GPU — even another Jetson with a different CC.

### Video backends: GStreamer vs V4L2

On a standard Linux desktop, `cv2.VideoCapture(0)` talks directly to the kernel's
V4L2 (Video4Linux2) camera driver. You can also explicitly force it with
`cv2.VideoCapture(0, cv2.CAP_V4L2)`.

On JetPack, OpenCV is compiled with GStreamer support and defaults to it. **GStreamer**
is a multimedia pipeline framework — it handles format negotiation, decoding, and
color conversion between camera hardware and your application. NVIDIA builds its camera
drivers to expose themselves through GStreamer.

When you force `CAP_V4L2` on JetPack, OpenCV bypasses GStreamer and talks raw to the
kernel driver. The BRIO outputs `yuv422_yuy2` format natively. Without GStreamer's
conversion pipeline, you get unprocessed frames — OpenCV receives them but can't
display them as RGB, which shows up as a black image.

The calibration script explicitly does NOT pass `CAP_V4L2`:
```python
cap = cv2.VideoCapture(args.device)       # GStreamer path — correct on JetPack
# NOT: cv2.VideoCapture(args.device, cv2.CAP_V4L2)  # bypasses GStreamer → black frames
```

You'll also see this in v4l2_camera's output: `"Image encoding not same as requested
output, performing slow conversion: yuv422_yuy2 => rgb8"`. That's GStreamer doing the
conversion in the ROS node. It's doing exactly what it should.

### JetPack and the software stack

**JetPack** is NVIDIA's Linux distribution for Jetson devices. Version 7.2 (what we
have) includes:
- Ubuntu 24.04
- CUDA 13.2
- TensorRT 10.16.2
- cuDNN (GPU-accelerated deep learning primitives)

These are all pre-installed at the OS level — they're not in the Python venv. The
venv uses `--system-site-packages` precisely so that TensorRT's Python bindings
(which live in system Python) are visible inside the venv without reinstalling them.

---

## 11. The Full Pipeline

Here is exactly what happens from "camera captures a frame" to "a point cloud grows":

```
USB Camera (physical photons hitting sensor)
        │
        │ MJPEG or raw frame (USB 2.0)
        ▼
v4l2_camera_node
  Reads the V4L2 device (/dev/video0)
  Publishes:
    /image_raw         (sensor_msgs/Image, bgr8, 640×480)
    /camera_info       (K matrix, distortion)
        │
        ▼
depth_anything_node (our node — backends.py + node.py)
  On each /image_raw message:
    1. Preprocess:
       - Convert BGR → RGB (numpy)
       - Resize to 364×364
       - ImageNet normalize
       - Shape: (1, 3, 364, 364) float32
    2. H2D copy: input → GPU buffer
    3. TRT execute_async_v3: run 70fps inference
    4. Stream sync: wait for GPU
    5. D2H copy: output → CPU buffer
    6. Postprocess:
       - Reshape output to 364×364 float32
       - Resize back to 640×480 (bilinear)
       - Values: metres (0–20)
  Publishes:
    /depth/image       (sensor_msgs/Image, 32FC1, 640×480)
    /depth/camera_info (same K matrix as input, updated header)
        │
        ▼
rgbd_odometry (rtabmap_ros package)
  Receives (approx-synced): /image_raw + /camera_info
                           + /depth/image + /depth/camera_info
  For each synced frame pair:
    - Detects and matches visual features between consecutive frames
    - Uses depth to compute 3D positions of matched features
    - Solves for camera motion (6-DOF: x, y, z, roll, pitch, yaw)
  Publishes:
    /odom              (nav_msgs/Odometry — camera pose estimate)
    /tf                (TF tree update: camera → odom → map)
        │
        ▼
rtabmap (rtabmap_slam package)
  Receives: RGB + depth + /odom
  - Projects depth pixels to 3D using K matrix + current pose
  - Colors each 3D point with the corresponding RGB pixel
  - Adds colored points to the global map
  - Runs loop closure detector on every keyframe
  - Optimizes pose graph when loop closure found
  Publishes:
    /rtabmap/cloud_map  (accumulated colored point cloud)
        │
        ▼
rtabmap_viz
  Renders /rtabmap/cloud_map in a 3D viewer
  Shows the camera trajectory and the growing colored reconstruction
  This is what you see on screen — the map being built in real time
```

### Why fps matters

The depth model runs at ~70 fps internally, but the whole pipeline is rate-limited
by the slowest component. On Phase 1a (laptop), the PyTorch backend was ~1.5 fps
and the entire pipeline ran at that rate.

On the Jetson the answer turned out to be none of the obvious candidates. Measured
2026-07-27, each part in isolation:

| Component | Rate |
|---|---|
| Camera alone (848×480, USB 2.0) | 27 Hz |
| Depth node compute ceiling | 32 fps (31.5 ms/frame) |
| TRT inference alone | ~47 fps (21.3 ms) |
| **Pipeline, all five processes** | **7–10 Hz** |
| **Pipeline, composed (fresh map, clocks pinned)** | **27.7 Hz** |

Every part was fast. The assembly was slow. The whole gap was inter-process message
transport — see *"What a topic actually costs"* in section 8. Composing the SLAM nodes
into one process took it to **20.9 Hz** without touching the model, the engine, or the
camera — and 27.7 Hz once the clocks were pinned and the socket buffers raised.

Two things worth stealing from this:

1. **Profile the parts before optimizing the whole.** Had we started by trying to speed
   up the depth network — quantizing to INT8, shrinking the input — we'd have spent days
   attacking a component that was never the constraint.
2. **Inference time is not the same as the benchmark.** Phase 0 measured 14.16 ms for
   the engine on an idle Jetson. In situ, with the rest of the pipeline competing for
   CPU and the governor set to `schedutil` rather than pinned clocks, the same engine
   measures 21.3 ms. Benchmarks are an upper bound, not a prediction.

### What changes in Phase 2

In Phase 2, when SPOT is involved:
- The camera becomes the Arducam IMX462 (drop-in replacement for the USB webcam —
  same UVC protocol, just a different device file)
- `rgbd_odometry` gets replaced by **SPOT's body odometry** from `spot_ros2`.
  SPOT's IMU + leg kinematics give much more accurate pose than visual odometry,
  especially in low-texture or dark environments.
- A **scale recovery** step ties the depth output to SPOT's known real-world motion —
  ensuring the map's metric scale is anchored to actual metres, not just the model's
  training distribution.

---

*Updated as new concepts are encountered in the project.*
