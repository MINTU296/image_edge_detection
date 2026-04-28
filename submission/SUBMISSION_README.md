# GPU Edge Detection — Submission Package

This package is evidence that the code in `run_edge_detection.py` runs end-to-end
on the GPU and processes both small and large data.

## Layout

```
submission/
├── run_edge_detection.py     # Runnable port of the Colab notebook (no Colab deps)
├── _prepare_inputs.py        # Stages inputs at 256, 1024, 4096 resolutions
├── inputs/                   # Before images (clearly prefixed `before_`)
│   ├── before_01_original_BenTennyson.jpg          (720 x 540)
│   ├── before_02_BenTennyson_256x256.png           (256 x 256)
│   ├── before_02_BenTennyson_1024x1024.png         (1,024 x 1,024)
│   └── before_02_BenTennyson_4096x4096.png         (4,096 x 4,096 — 16.7 MP)
├── outputs/                  # Sobel edge-magnitude results (`after_*_edges.png`)
│   ├── after_01_original_BenTennyson_edges.png
│   ├── after_02_BenTennyson_256x256_edges.png
│   ├── after_02_BenTennyson_1024x1024_edges.png
│   └── after_02_BenTennyson_4096x4096_edges.png
└── logs/
    └── log.txt               # Execution log with TF version, GPU, timings, throughput
```

## How it ran

```
python run_edge_detection.py inputs outputs logs/log.txt
```

The log records the host, OS, Python/TensorFlow versions, the detected GPU
(`Apple M4 Pro` Metal device), the input shape and pixel count for each image,
the GPU convolution time after warm-up, and the throughput in megapixels/sec.

## Demonstration of scale

The largest image is **4,096 × 4,096 (16.7 megapixels)**, processed in ~60 ms on
the GPU at ~280 Mpix/s. Aggregate throughput across the four-image batch is
recorded in the log.

## Reproducing

```
pip install tensorflow-macos tensorflow-metal pillow numpy   # macOS Apple Silicon
# or: pip install tensorflow pillow numpy                    # CUDA GPU box
python _prepare_inputs.py "<path-to-source-image>" inputs
python run_edge_detection.py inputs outputs logs/log.txt
```
