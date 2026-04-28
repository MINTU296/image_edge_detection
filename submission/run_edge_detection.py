"""GPU-accelerated Sobel edge detection (TensorFlow).

Runnable port of edge_detection_gpu_tensorflow.ipynb.ipynb.
Processes any number of input images, saves before/after pairs, and writes
a verbose execution log so the artifact itself is proof of GPU execution.

Usage:
    python run_edge_detection.py <input_dir> <output_dir> <log_file>
"""

import os
import sys
import time
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


def build_kernels():
    sobel_x = tf.constant([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]], dtype=tf.float32)
    sobel_y = tf.constant([[-1.0, -2.0, -1.0],
                           [ 0.0,  0.0,  0.0],
                           [ 1.0,  2.0,  1.0]], dtype=tf.float32)
    return tf.reshape(sobel_x, [3, 3, 1, 1]), tf.reshape(sobel_y, [3, 3, 1, 1])


SOBEL_X, SOBEL_Y = build_kernels()


@tf.function
def sobel_edges(x):
    gx = tf.nn.conv2d(x, SOBEL_X, strides=[1, 1, 1, 1], padding='SAME')
    gy = tf.nn.conv2d(x, SOBEL_Y, strides=[1, 1, 1, 1], padding='SAME')
    return tf.sqrt(tf.square(gx) + tf.square(gy))


def load_image_as_tensor(path):
    img = Image.open(path).convert('L')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr, tf.convert_to_tensor(arr[None, ..., None], dtype=tf.float32)


def save_edge_image(edges_tf, out_path):
    edges_np = tf.squeeze(edges_tf).numpy()
    edges_np = edges_np / (edges_np.max() + 1e-8)
    edges_uint8 = np.clip(edges_np * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(edges_uint8).save(out_path)
    return edges_uint8.shape


def gpu_available():
    return len(tf.config.list_physical_devices('GPU')) > 0


def process_one(input_path, output_path, device, log):
    arr, tensor = load_image_as_tensor(input_path)
    h, w = arr.shape

    with tf.device(device):
        # Warm-up so the timed run reflects steady-state GPU performance.
        _ = sobel_edges(tensor)
        start = time.perf_counter()
        edges = sobel_edges(tensor)
        _ = edges.numpy()  # force device sync
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    out_shape = save_edge_image(edges, output_path)
    pixels = h * w
    throughput = pixels / (elapsed_ms / 1000.0) / 1e6  # Mpix/s

    log(f"  input    : {input_path}")
    log(f"  size     : {w} x {h} ({pixels:,} pixels)")
    log(f"  device   : {device}")
    log(f"  conv ms  : {elapsed_ms:.3f}")
    log(f"  Mpix/s   : {throughput:,.1f}")
    log(f"  output   : {output_path}  shape={out_shape}")
    log("")
    return elapsed_ms, pixels


def main():
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    log_path = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("GPU Edge Detection - Execution Log")
    log("==================================")
    log(f"Started (UTC)    : {datetime.now(timezone.utc).isoformat()}")
    log(f"Host             : {platform.node()}")
    log(f"Platform         : {platform.system()} {platform.release()} ({platform.machine()})")
    log(f"Python           : {sys.version.split()[0]}")
    log(f"TensorFlow       : {tf.__version__}")
    log(f"NumPy            : {np.__version__}")
    log(f"Pillow           : {Image.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    log(f"GPUs detected    : {len(gpus)}")
    for g in gpus:
        log(f"  - {g.name} ({g.device_type})")
    device = '/GPU:0' if gpus else '/CPU:0'
    log(f"Selected device  : {device}")
    log("")

    inputs = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    log(f"Found {len(inputs)} input image(s) in {input_dir}")
    log("")

    total_ms = 0.0
    total_pixels = 0
    for i, in_path in enumerate(inputs, 1):
        stem = in_path.stem
        if stem.startswith("before_"):
            stem = stem[len("before_"):]
        out_name = f"after_{stem}_edges.png"
        out_path = output_dir / out_name
        log(f"[{i}/{len(inputs)}] {in_path.name}")
        elapsed_ms, pixels = process_one(str(in_path), str(out_path), device, log)
        total_ms += elapsed_ms
        total_pixels += pixels

    log("Summary")
    log("-------")
    log(f"Images processed : {len(inputs)}")
    log(f"Total pixels     : {total_pixels:,}")
    log(f"Total conv time  : {total_ms:.3f} ms")
    if total_ms > 0:
        log(f"Aggregate Mpix/s : {(total_pixels / (total_ms / 1000.0)) / 1e6:,.1f}")
    log(f"Finished (UTC)   : {datetime.now(timezone.utc).isoformat()}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
