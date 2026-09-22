"""Convert ArcGIS frames to SAM 3.1 model input.

SAM 3.1 preprocessing (upstream `sam3/model/io_utils.py`, async loader
`_transform_frame` and the PIL-list path of `load_resource_as_video_frames`):

    frame (H, W, 3) uint8 RGB
      -> bicubic resize to (image_size, image_size)   [no aspect preservation]
      -> / 255
      -> (x - 0.5) / 0.5
      -> float16 storage, cast to float32 on the GPU right before the backbone

Output masks are later resized back to the original (H, W) by the tracker
(`_get_orig_video_res_output`), so boxes derived from them are already in
original pixel space.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SIZE = 1008
IMG_MEAN = (0.5, 0.5, 0.5)
IMG_STD = (0.5, 0.5, 0.5)

VALID_CHANNEL_ORDERS = ("RGB", "BGR")


class FrameFormatError(ValueError):
    pass


def normalize_frame_layout(frame, channel_order="RGB"):
    """Return the frame as (H, W, 3) uint8 RGB numpy array.

    Accepts HWC or CHW arrays with 1, 3 or 4 channels, uint8 or float
    (float in 0..1 or 0..255). Raises FrameFormatError for anything else.
    """
    if frame is None:
        raise FrameFormatError("frame is None")
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise FrameFormatError(f"expected a 2-D or 3-D frame, got shape {arr.shape}")

    # CHW -> HWC when the leading axis looks like a channel axis
    if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[0] in (1, 3, 4) and arr.shape[2] in (1, 3, 4) and arr.shape[0] < arr.shape[2]:
        # ambiguous tiny frames: prefer HWC (ArcGIS convention)
        pass

    c = arr.shape[2]
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif c == 4:
        arr = arr[:, :, :3]
    elif c != 3:
        raise FrameFormatError(f"expected 1, 3 or 4 channels, got {c} (shape {arr.shape})")

    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if np.isfinite(a).all() and a.max(initial=0.0) <= 1.0:
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)

    channel_order = (channel_order or "RGB").upper()
    if channel_order not in VALID_CHANNEL_ORDERS:
        raise FrameFormatError(f"channel_order must be one of {VALID_CHANNEL_ORDERS}")
    if channel_order == "BGR":
        arr = arr[:, :, ::-1]

    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        raise FrameFormatError(f"frame too small: {h}x{w}")
    return np.ascontiguousarray(arr)


def frame_to_model_tensor(frame, device, image_size=IMAGE_SIZE, channel_order="RGB"):
    """ArcGIS frame -> (tensor (1, 3, S, S) float32 on `device`, H, W).

    Resizing runs on the GPU (bicubic, align_corners=False) to match the
    upstream async video loader.
    """
    rgb = normalize_frame_layout(frame, channel_order=channel_order)
    h, w = rgb.shape[:2]
    t = torch.from_numpy(rgb).to(device=device, non_blocking=True)
    t = t.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W) 0..255
    if (h, w) != (image_size, image_size):
        t = F.interpolate(t, size=(image_size, image_size), mode="bicubic", align_corners=False)
        t = t.clamp_(0.0, 255.0)
    t = t.half()  # fp16 storage precision, as upstream
    t = t / 255.0
    mean = torch.tensor(IMG_MEAN, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMG_STD, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.float(), h, w
