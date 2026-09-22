"""SAM 3.1 runtime for the text-prompt video-tracking toolbox.

Two model instances are loaded, both ONCE, both from the same local SAM 3.1
    multiplex checkpoint selected from a compatible local DLPK:

  * `self.session` -- a pool of sam31_runtime streaming sessions sharing
    one tracker model and one backbone evaluation per frame.
    Used for every frame: cheap, no text encoder / grounding detector
    involved, includes the occlusion-safe "held box" behaviour already
    required by the native ArcGIS host.

  * `self.detector` -- the FULL SAM 3.1 multiplex predictor (grounding
    detector + tracker), built with the vendored
    `sam3.model_builder.build_sam3_multiplex_video_predictor`. Used ONLY
    when a text-prompt detection is requested (frame 0, then every
    DETECTION_INTERVAL frames). Its own frame-by-frame tracking/session
    machinery is never used for continuous tracking -- that stays on
    `self.session` -- so the (expensive) grounding detector never runs on
    frames the caller did not ask for.

Both models are cached. The pipeline decodes sequentially and hands the
detector a two-frame window. Surviving objects retain their SAM memory;
replacement objects use unused multiplex slots or a new lightweight session
because upstream tombstones cannot safely be reused. Vendored SAM is unchanged.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Runtime code is release-relative; selected DLPKs supply assets only.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLPK_PKG_ROOT = os.path.join(PROJECT_ROOT, "SAM31_ObjectTracker")
if DLPK_PKG_ROOT not in sys.path:
    sys.path.insert(0, DLPK_PKG_ROOT)
for _name in ('sam3', 'sam31_runtime'):
    _loaded = sys.modules.get(_name)
    if _loaded is not None:
        _file = getattr(_loaded, '__file__', None)
        if _file and not Path(_file).resolve().is_relative_to(Path(DLPK_PKG_ROOT).resolve()):
            raise RuntimeError('A different SAM runtime is already loaded. Restart ArcGIS Pro before using this release.')

import torch  # noqa: E402
import cv2  # noqa: E402

from sam31_runtime.logging_utils import get_logger  # noqa: E402
from sam31_runtime.model_loader import get_tracker, ModelLoadError  # noqa: E402
from sam31_runtime.sam31_session import SAM31StreamingSession  # noqa: E402
from sam31_runtime.arcgis_boxes import sanitize_box  # noqa: E402

from video_reader import VideoReader, normalize_path, probe_video  # noqa: E402
from streaming_detector_state import (  # noqa: E402
    CURRENT, StreamingFrameStore, build_streaming_state, register_session,
)

DEFAULT_CHECKPOINT = os.path.join(DLPK_PKG_ROOT, "model", "sam3.1_multiplex.pt")
DEFAULT_BPE = os.path.join(DLPK_PKG_ROOT, "model", "bpe_simple_vocab_16e6.txt.gz")


class VideoMemoryError(RuntimeError):
    """The legacy full-video loader ran out of memory preloading frames."""


def _find_memory_error(exc: BaseException) -> Optional[BaseException]:
    """Return the underlying allocation failure in this exception chain, if any.

    The vendored loader catches a NumPy allocation failure and re-raises it as
    NotImplementedError("Only video files and image folders are supported"),
    which is badly misleading: the file decoded fine, the machine simply could
    not hold every frame at once. Walk the chain to find the real cause so the
    message we show quotes that instead of the misleading wrapper.
    """
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryError) or type(current).__name__ == "_ArrayMemoryError":
            return current
        if "unable to allocate" in str(current).lower():
            return current
        current = current.__cause__ or current.__context__
    return None


def _is_memory_error(exc: BaseException) -> bool:
    return _find_memory_error(exc) is not None


def _legacy_loader_error(video_path: str, exc: BaseException) -> Exception:
    """Turn a full-preload failure into an accurate, actionable error.

    Quotes the underlying allocation failure, never the vendored loader's
    misleading "Only video files and image folders are supported" wrapper --
    a valid .ts must never be reported as an unsupported format.
    """
    root = _find_memory_error(exc)
    if root is None:
        return exc
    return VideoMemoryError(
        f"The video was decoded successfully, but the legacy full-video loader "
        f"attempted to preload all frames into memory and ran out of RAM "
        f"({os.path.basename(video_path)}). The video format is fine; the "
        f"machine cannot hold every decoded frame at once. Streaming video "
        f"processing should be used: call "
        f"SAM31VideoRuntime.open_video_streaming() (the default in pipeline.py) "
        f"instead of open_video_for_detection(). Allocation failure: {root}")


@dataclass
class Detection:
    """One text-prompt detection candidate, in pixel coordinates."""

    box: List[float]      # [x1, y1, x2, y2]
    score: float
    prompt: str
    mask: Optional[np.ndarray] = None  # current detection pass only


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int


class FrameReader:
    """Sequential RGB frame reader -- a thin adapter over video_reader.VideoReader.

    Tracking and export both need raw (H, W, 3) uint8 RGB frames; the detector
    keeps a resized, normalised float16 copy for its own use and does not
    expose original-resolution frames. Control flow only ever asks for frame t
    after frame t - 1 (matching the ArcGIS track(frame) call pattern the
    tracker session expects), so decoding is strictly forward-only: no
    seeking, no re-decoding, the file is read start to end exactly once.

    All container/codec handling now lives in video_reader, which treats the
    reported frame count as an estimate rather than a fact -- see that module
    for the formats where it is measurably wrong. `frame_count` here is that
    estimate and may be None; never use it as a decode-loop bound, read until
    read() returns None instead.
    """

    def __init__(self, path: str, probe=None, logger=None):
        self._reader = VideoReader(path, probe=probe, logger=logger)
        self.probe = self._reader.open()
        self.path = self._reader.path
        self.width = self._reader.width
        self.height = self._reader.height
        self.fps = self._reader.fps or 30.0
        self.frame_count = self._reader.frame_count_estimate
        self.next_index = 0

    def info(self) -> VideoInfo:
        return VideoInfo(self.path, self.width, self.height, self.fps, self.frame_count or 0)

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame as RGB uint8, or None at end of stream."""
        frame = self._reader.read_frame()
        if frame is None:
            return None
        self.next_index = self._reader.current_frame + 1
        return frame

    @property
    def current_timestamp(self) -> float:
        """Timestamp of the frame last returned, in seconds."""
        return self._reader.current_timestamp

    @property
    def container_timestamp(self) -> Optional[float]:
        """Source container PTS of the last frame, or None when unusable."""
        return self._reader.container_timestamp

    def close(self):
        self._reader.close()


class SAM31VideoRuntime:
    """Owns both SAM 3.1 model instances and exposes a tiny detect/track API."""

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT, bpe_path: str = DEFAULT_BPE,
                 device: str = "cuda", multiplex_count: int = 16, confidence_threshold: float = 0.35,
                 keep_window: int = 64, tracker_channel_order: str = "RGB",
                 detector_max_num_objects: int = 32, offload_detector_video_to_cpu: bool = True,
                 async_loading_frames: bool = True, swap_tracker_to_cpu_during_detect: bool = True,
                 logger=None, debug: bool = False):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.keep_window = keep_window
        self.channel_order = tracker_channel_order
        self.detector_max_num_objects = detector_max_num_objects
        self.offload_detector_video_to_cpu = offload_detector_video_to_cpu
        self.async_loading_frames = async_loading_frames
        self.swap_tracker_to_cpu_during_detect = swap_tracker_to_cpu_during_detect
        self.log = logger or get_logger(debug)

        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        self.multiplex_count = multiplex_count

        self.tracking_model = None
        self.session = None
        self.detector = None
        self._detector_session_id: Optional[str] = None
        self._video_info: Optional[VideoInfo] = None
        # Streaming detection (bounded memory) -- see streaming_detector_state.
        self._frame_store: Optional[StreamingFrameStore] = None
        self._streaming = False

        self.load_times: Dict[str, float] = {}

    # ------------------------------------------------------------------ load
    def load(self):
        """Build both models once. Safe to call again: both loaders cache."""
        if self.is_loaded:
            return
        if not os.path.isfile(self.checkpoint_path):
            raise ModelLoadError(f"checkpoint not found: {self.checkpoint_path}")
        if not torch.cuda.is_available():
            raise ModelLoadError("SAM 3.1 requires a CUDA GPU (same requirement as the DLPK)")

        t0 = time.time()
        # Same call the DLPK's initialize() makes -- process-wide cached, so a
        # second SAM31VideoRuntime in the same process reuses this model.
        self.tracking_model = get_tracker(
            self.checkpoint_path, device=self.device, multiplex_count=self.multiplex_count,
            compile_model=False, logger=self.log,
        )
        from streaming_session_pool import StreamingSessionPool
        self.session = StreamingSessionPool(
            self.tracking_model, device=self.device, confidence_threshold=self.confidence_threshold,
            keep_window=self.keep_window, channel_order=self.channel_order, logger=self.log,
        )
        self.load_times["tracker_s"] = time.time() - t0
        self.log.info("tracking model ready in %.1fs", self.load_times["tracker_s"])

        t0 = time.time()
        from detector_cache import get_detector

        if self.swap_tracker_to_cpu_during_detect:
            self.tracking_model.to("cpu")
        try:
            self.detector = get_detector(
                checkpoint_path=self.checkpoint_path, bpe_path=self.bpe_path, logger=self.log,
                max_num_objects=self.detector_max_num_objects, multiplex_count=self.multiplex_count,
                use_fa3=False, use_rope_real=True, compile=False, warm_up=False,
                async_loading_frames=self.async_loading_frames,
            )
            self.detector.model.to("cpu" if self.swap_tracker_to_cpu_during_detect else self.device)
        finally:
            self.tracking_model.to(self.device)
        self.load_times["detector_s"] = time.time() - t0
        self.log.info("text-prompt detector ready in %.1fs", self.load_times["detector_s"])

        if self.swap_tracker_to_cpu_during_detect:
            # Steady state: only the tracker (~2.3 GB) stays resident on the GPU; the much
            # larger detector (~5.3 GB) is parked on CPU until a detection is actually
            # requested. See `_detection_window` for why both models being resident at once
            # is unsafe on a 12 GB GPU (measured: not an OOM error, a silent multi-minute
            # stall from Windows/WDDM paging VRAM to system memory).
            self.detector.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log.info("detector parked on CPU (moved to GPU only during detect())")
        if torch.cuda.is_available():
            self.log.info("GPU memory after loading both models: %.0f MB allocated",
                          torch.cuda.memory_allocated() / 2**20)

    @property
    def is_loaded(self):
        return self.tracking_model is not None and self.detector is not None

    # ------------------------------------------------------------------ video
    def open_video_for_detection(self, video_path: str) -> VideoInfo:
        """Load the video ONCE into the detector's own frame store.

        Bypasses Sam3BasePredictor.start_session and calls
        self.detector.model.init_state directly, then registers the session
        by hand. This works around a reproduced upstream bug (documented as
        "U1" in SAM31_ObjectTracker/PATCHES.md): start_session always
        forwards offload_state_to_cpu= but
        Sam3MultiplexTrackingWithInteractivity.init_state does not accept
        that keyword. No vendored code is modified; this is a call-site
        workaround, the same one used by
        tests/test_sam31_upstream_equivalence.py.
        """
        if self.detector is None:
            raise RuntimeError("call .load() first")
        if self._detector_session_id is not None:
            self.close_video()
        # SAM 3.1's own loader decodes with cv2 as well, so it needs the same
        # absolutised path -- manifest inputs do not resolve their segments
        # from a relative path (see video_reader.normalize_path).
        video_path = normalize_path(video_path)
        try:
            state = self.detector.model.init_state(
                resource_path=video_path,
                offload_video_to_cpu=self.offload_detector_video_to_cpu,
                async_loading_frames=self.async_loading_frames,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised below with a usable message
            raise _legacy_loader_error(video_path, exc) from exc
        sid = str(uuid.uuid4())
        self.detector._all_inference_states[sid] = {
            "state": state, "session_id": sid, "start_time": time.time(), "last_use_time": time.time(),
        }
        self._detector_session_id = sid
        self._video_info = VideoInfo(
            video_path, state["orig_width"], state["orig_height"],
            fps=0.0, frame_count=state["num_frames"],
        )
        self.log.info("detector: video loaded once (%d frames, %dx%d, offload_to_cpu=%s)",
                      state["num_frames"], state["orig_width"], state["orig_height"],
                      self.offload_detector_video_to_cpu)
        return self._video_info

    def open_video_streaming(self, video_path: str) -> VideoInfo:
        """Open the video for detection WITHOUT decoding it into memory.

        The detector is given a 2-frame window (see streaming_detector_state)
        instead of the whole video, so RAM is independent of duration and the
        total frame count -- unknowable for MPEG-TS -- is never needed. The
        caller feeds frames via `set_detection_window` as it decodes them
        sequentially; nothing is re-decoded or seeked here.
        """
        if self.detector is None:
            raise RuntimeError("call .load() first")
        if self._detector_session_id is not None:
            self.close_video()

        video_path = normalize_path(video_path)
        probe = probe_video(video_path)
        if not probe.width or not probe.height:
            raise ModelLoadError(
                f"could not determine frame size for {video_path}; cannot set up detection")

        self._frame_store = StreamingFrameStore(
            self.detector.model.image_size,
            img_mean=self.detector.model.image_mean,
            img_std=self.detector.model.image_std,
            logger=self.log,
        )
        state = build_streaming_state(self.detector.model, self._frame_store,
                                      orig_height=probe.height, orig_width=probe.width)
        self._detector_session_id = register_session(self.detector, state)
        self._streaming = True
        self._video_info = VideoInfo(video_path, probe.width, probe.height,
                                     fps=probe.fps or 0.0,
                                     frame_count=probe.frame_count_estimate or 0)
        self.log.info(
            "detector opened in STREAMING mode (%dx%d): %d-frame window, no full-video preload",
            probe.width, probe.height, len(self._frame_store))
        return self._video_info

    def set_detection_window(self, frames_rgb, frame_numbers=None):
        """Install the frames the next detect() call may read.

        frames_rgb[0] is the frame being detected on; frames_rgb[1] is the
        following frame, which SAM 3.1 also reads (measured). At end of video a
        single frame is accepted and the store clamps.
        """
        if self._frame_store is None:
            raise RuntimeError("call .open_video_streaming() first")
        self._frame_store.set_window(frames_rgb, frame_numbers)

    def close_video(self):
        if self._frame_store is not None:
            self._frame_store.clear()
            self._frame_store = None
        self._streaming = False
        if self._detector_session_id is not None:
            try:
                self.detector.close_session(self._detector_session_id)
            except Exception:  # noqa: BLE001 - best effort cleanup
                self.log.warning("error closing detector session", exc_info=True)
            self._detector_session_id = None
        if self.session is not None:
            self.session.close()

    # ------------------------------------------------------------------ detect
    def _move(self, module, device: str, tag: str):
        t0 = time.time()
        module.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.log.debug("moved %s to %s in %.0f ms", tag, device, (time.time() - t0) * 1000.0)

    class _DetectionWindow:
        """Keep only the active model resident on the GPU during detection.

        Both models remain loaded. Moving the idle model to CPU avoids WDDM
        paging when concurrent residency exceeds GPU capacity. This changes
        residency only; weights and tracking state are not reconstructed.
        """

        def __init__(self, runtime: "SAM31VideoRuntime"):
            self.runtime = runtime

        def __enter__(self):
            r = self.runtime
            if r.swap_tracker_to_cpu_during_detect:
                r._move(r.tracking_model, "cpu", "tracker")
                r._move(r.detector.model, r.device, "detector")
            return self

        def __exit__(self, *exc):
            r = self.runtime
            if r.swap_tracker_to_cpu_during_detect:
                r._move(r.detector.model, "cpu", "detector")
                r._move(r.tracking_model, r.device, "tracker")
            return False

    def detect(self, frame_idx: int, text_prompt: str,
               prob_thresh: Optional[float] = None) -> Tuple[List[Detection], float]:
        """Run ONE text-prompt grounding pass on frame_idx of the loaded video.

        Uses add_prompt directly (not propagate_in_video), so this never
        touches frames outside frame_idx and never runs the detector on
        frames the caller did not ask for. add_prompt resets its own
        internal per-prompt state before running (upstream behaviour), so
        repeated calls at different frame indices/prompts are independent.

        GPU residency is swapped for the duration of the call -- see
        `_DetectionWindow` -- and always restored, even on an exception.
        """
        if self._detector_session_id is None:
            raise RuntimeError("call .open_video_for_detection() first")
        thresh = self.confidence_threshold if prob_thresh is None else prob_thresh
        # In streaming mode the model state holds only the current 2-frame
        # window, so it is always asked about the local index; frame_idx stays
        # the caller's real frame number for logging and result bookkeeping.
        request_index = CURRENT if self._streaming else frame_idx
        t0 = time.time()
        with self._DetectionWindow(self):
            response = self.detector.handle_request(dict(
                type="add_prompt", session_id=self._detector_session_id, frame_index=request_index,
                text=text_prompt, output_prob_thresh=thresh,
            ))
        latency_ms = (time.time() - t0) * 1000.0
        out = response["outputs"]
        probs = out["out_probs"]
        boxes_xywh = out["out_boxes_xywh"]
        W, H = self._video_info.width, self._video_info.height
        detections: List[Detection] = []
        for i in range(len(probs)):
            score = float(probs[i])
            if score < thresh:
                continue
            x, y, w, h = (float(v) for v in boxes_xywh[i])
            box = sanitize_box([0, x * W, y * H, (x + w) * W, (y + h) * H], W, H)
            if box is None:
                continue
            mask = out.get("out_binary_masks")
            mask = np.array(mask[i], dtype=bool, copy=True) if mask is not None else None
            if mask is not None and not mask.any():
                continue
            if mask is not None:
                # Upstream boxes may precede mask non-overlap processing. Match
                # and initialize using the actual final mask's pixel extent.
                from track_quality import mask_box
                box = [0] + mask_box(mask)
            detections.append(Detection(box=box[1:], score=score, prompt=text_prompt, mask=mask))
        self.log.info("detect(frame=%d, prompt=%r): %d candidate(s) >= %.2f in %.0f ms",
                      frame_idx, text_prompt, len(detections), thresh, latency_ms)
        return detections, latency_ms

    def warm_up_detector(self) -> float:
        """Prepare lazy detector kernels once using the current streaming window.

        A threshold above 1 runs inference without introducing tracks. The
        cached predictor records successful preparation for subsequent runs.
        Cold preparation can take several minutes on the tested environment.
        """
        if getattr(self.detector, "_sam31_warmed_up", False):
            self.log.info("detector already warmed up (cached model)")
            return 0.0
        _, latency_ms = self.detect(0, "__sam31_warmup_probe__", prob_thresh=2.0)
        self.detector._sam31_warmed_up = True
        self.log.info("detector warm-up (one-time init) took %.0f ms", latency_ms)
        return latency_ms

    # -------------------------------------------------- tracking (delegates to sam31_runtime)
    def start_tracking(self, frame_h: int, frame_w: int):
        self.session.start(frame_h, frame_w)

    def init_tracks(self, frame: np.ndarray, boxes_with_ids: List[List[float]], initial_masks=None):
        """boxes_with_ids: rows of [track_id, x1, y1, x2, y2]."""
        if initial_masks is None:
            return self._checked_session_call(self.session.add_objects, frame, boxes_with_ids)
        return self._checked_session_call(
            lambda f, b: self.session.add_objects(f, b, initial_masks=initial_masks), frame, boxes_with_ids)

    def track_frame(self, frame: np.ndarray):
        return self._checked_session_call(self.session.step, frame)

    def _checked_session_call(self, method, *args):
        # Native Pro requires held rows on failure. The batch toolbox must not
        # export those rows as successful observations and continue a broken run.
        errors = self.session.stats["errors"]
        result = method(*args)
        if self.session.stats["errors"] != errors:
            raise RuntimeError("SAM tracking failed; see the session error and trace above.")
        return result

    def shutdown(self):
        self.close_video()

