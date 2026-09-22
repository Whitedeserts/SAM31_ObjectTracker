# Environment

Use ArcGIS Pro 3.7 and its matching Esri deep-learning libraries on Windows.
Recorded versions: Python 3.13; PyTorch 2.9.1 with CUDA 12.9; torchvision 0.25.0.
Other combinations require validation. No environment lockfile is provided because
ArcGIS manages these dependencies; do not run a general pip upgrade in that environment.

A compatible NVIDIA CUDA GPU/driver is required. The test system had approximately
12 GB GPU memory; this is not a guaranteed minimum. Object count, frame size,
category count and session turnover affect memory/time. CPU-only inference is unsupported.
Allow at least 3.3 GiB per extracted model cache, additional temporary extraction
space, the DLPK download, and sufficient disk space for outputs. Multiple distinct
DLPK fingerprints can create separate caches even when their checkpoints match.

External imports: arcpy, torch, torchvision, numpy, pandas, scipy, Pillow, OpenCV,
timm, ftfy, regex, iopath, tqdm, einops and huggingface_hub. psutil provides memory
diagnostics. ffprobe is used for metadata probing; FFmpeg tools resolve from the
active environment or PATH. Optional Triton/FlashAttention paths are guarded in the
Windows runtime. Dependencies are not vendored or installed by this project.

Normal processing does not download models or rewrite the source video. First model
initialization can take longer than later runs. Separate detector and tracking
models are cached within the Python process; changing runtime versions requires
restarting ArcGIS Pro. Concurrent runs in one Python process are unsupported.
