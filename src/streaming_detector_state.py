"""Bounded two-frame detector state for sequential video processing.

The pinned detector reads the requested frame and its successor. Local indices
map onto the current decode window, including the final-frame fallback. Bounding
num_frames also bounds detector metadata and prompt-reset work. Tracking state is
maintained separately; no complete-video tensor or frame history is retained."""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

import cv2
import numpy as np
import torch

# Local index the detector is always asked about. The store maps it to the
# caller's real frame; the real frame number never enters the model state.
CURRENT = 0
WINDOW = 2          # measured requirement: frames N and N+1
DUMMY_RESOURCE = "<load-zero-video-1>"   # 1-frame scaffold, ~6 MB, not the video


class StreamingFrameStore:
    """Serves the detector a 2-frame window instead of the whole video.

    Deliberately NOT a `collections.abc.Sequence`: `recursive_to` in
    sam3_multiplex_tracking iterates anything that registers as a Sequence,
    which would materialise every frame and defeat the entire purpose.
    Supports the index types the vendored code uses (int, 0-d/1-d tensors,
    lists) because `img_batch.tensors[...]` is indexed directly in places.
    """

    def __init__(self, image_size: int,
                 img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5),
                 window: int = WINDOW, logger=None):
        self.image_size = int(image_size)
        self.window = int(window)
        self.log = logger
        # float16 mean/std viewed (3,1,1): matches the per-frame arithmetic of
        # load_video_frames_from_video_file_using_cv2 exactly (which subtracts a
        # float16 (1,3,1,1) from a float32 stack), so pixel values are bit-identical.
        self._mean = torch.tensor(img_mean, dtype=torch.float16).view(3, 1, 1)
        self._std = torch.tensor(img_std, dtype=torch.float16).view(3, 1, 1)
        self._slots: dict = {}
        self.source_frame_numbers: List[Optional[int]] = [None] * self.window

    # ------------------------------------------------------------------ fill
    def preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 (H,W,3) -> normalised CHW float32, as SAM 3.1 expects."""
        resized = cv2.resize(rgb, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_CUBIC)
        tensor = torch.from_numpy(resized.astype(np.float32)).permute(2, 0, 1)
        tensor -= self._mean
        tensor /= self._std
        return tensor

    def set_window(self, frames_rgb, source_frame_numbers=None):
        """Install the current window: frames_rgb[0] is the frame to detect on.

        frames_rgb may be short (1 item) at end of video; see __getitem__.
        """
        self._slots = {i: self.preprocess(frame)
                       for i, frame in enumerate(frames_rgb[:self.window])}
        numbers = list(source_frame_numbers or [])
        self.source_frame_numbers = (numbers + [None] * self.window)[:self.window]

    def clear(self):
        self._slots = {}

    @property
    def resident_frames(self) -> int:
        return len(self._slots)

    def approx_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._slots.values())

    # ----------------------------------------------------------------- index
    def __len__(self):
        return self.window

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item() if idx.numel() == 1 else idx.tolist()
        if isinstance(idx, np.ndarray):
            idx = idx.item() if idx.size == 1 else idx.tolist()
        if isinstance(idx, (list, tuple)):
            return torch.stack([self[i] for i in idx])
        if isinstance(idx, slice):
            return torch.stack([self[i] for i in range(*idx.indices(self.window))])

        i = int(idx)
        if not self._slots:
            raise RuntimeError(
                "StreamingFrameStore has no frames installed; call set_window() "
                "before running detection.")
        if i in self._slots:
            return self._slots[i]
        # Past end of video: the model reads N+1, which does not exist on the
        # final frame. Clamp to the newest frame, mirroring the vendored
        # detector's own `min(frame_idx_begin + rank, frame_idx_end - 1)`.
        newest = max(self._slots)
        if self.log:
            self.log.debug("frame %d beyond window; clamping to %d (end of video)", i, newest)
        return self._slots[newest]


def build_streaming_state(model, store: StreamingFrameStore,
                          orig_height: int, orig_width: int) -> dict:
    """Create a detector inference state backed by `store` rather than a video.

    The real `init_state` is used to build correct scaffolding (it sets up
    action history, tracker sub-states and everything else the request handlers
    expect) but is pointed at a 1-frame dummy so it never decodes the video.
    The per-frame structures are then rebuilt around the store by the vendored
    `_construct_initial_input_batch`, so none of that construction logic is
    duplicated here.
    """
    state = model.init_state(resource_path=DUMMY_RESOURCE,
                             offload_video_to_cpu=True, async_loading_frames=False)
    # Real source dimensions: output boxes are normalised against these, so they
    # must be the video's own size, not the dummy's.
    state["orig_height"] = int(orig_height)
    state["orig_width"] = int(orig_width)
    state["num_frames"] = len(store)
    model._construct_initial_input_batch(state, store)
    # num_frames/orig_* changed, so the tracker sub-state must be rebuilt from
    # them -- same call reset_state makes.
    if model.tracker.per_obj_inference:
        state["sam2_inference_states"] = [model._init_new_sam2_state(state)]
    else:
        state["sam2_inference_states"] = []
    state["is_image_only"] = False
    return state


def register_session(predictor, state: dict) -> str:
    """Register a hand-built state with the predictor and return its session id.

    Bypasses `start_session` for the documented upstream bug U1 (it always
    forwards `offload_state_to_cpu=`, which this model's `init_state` does not
    accept) -- the same call-site workaround already used elsewhere here.
    """
    session_id = str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state": state, "session_id": session_id,
        "start_time": time.time(), "last_use_time": time.time(),
    }
    return session_id
