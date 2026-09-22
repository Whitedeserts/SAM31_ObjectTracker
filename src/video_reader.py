"""Sequential OpenCV video decoding with ffprobe metadata validation.

Container frame counts are estimates used only for progress. Decode until EOF;
never use reported counts as iteration bounds. ffprobe supplies timing and KLV
metadata without rewriting the input. Detection and tracking use the same decoded
frames. Actual support depends on the codecs available in the active environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

# Quieten the FFmpeg decoder's per-frame chatter (e.g. "co located POCs
# unavailable" on TS files that start mid-GOP). Must be set before the first
# VideoCapture; harmless if OpenCV has already initialised.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


class UnsupportedVideoError(RuntimeError):
    """Raised when an input cannot be opened or decoded by the toolbox."""


# Extensions validated end-to-end through this reader. Keep in sync with
# USER_GUIDE.md and the toolbox parameter filter.
SUPPORTED_EXTENSIONS = [
    "mp4", "mov", "avi", "mkv", "ts", "m2ts", "mpg", "mpeg", "ps", "vob",
    "wmv", "h264", "h265", "m3u8", "mpd",
]

# Containers where random seeking is unreliable in this environment, so the
# pipeline must stay strictly sequential (it already is; this drives messaging).
SEQUENTIAL_ONLY_EXTENSIONS = {".ts", ".m2ts", ".h264", ".h265", ".m3u8"}

# KLV/MISB detection. An explicit KLV codec name or tag counts in any
# container. The generic names ("bin_data", "data") are what ffprobe reports
# for ANY opaque data track -- GoPro/DJI telemetry in MP4, for instance -- so
# they only count inside an MPEG transport stream, where such a track is
# almost always MISB metadata. Keeps the FMV warning specific to FMV inputs.
_KLV_CODEC_NAMES = {"klv", "smpte_klv"}
_GENERIC_DATA_CODEC_NAMES = {"bin_data", "data"}


def normalize_path(path: str) -> str:
    """Absolutise local paths before they reach any decoder.

    Manifest inputs resolve their segment references relative to the manifest,
    and the FFmpeg demuxer resolves them against the process working directory
    rather than the manifest's own folder: measured here, the identical DASH
    manifest fails to open via a relative path and opens and decodes correctly
    via an absolute one. HLS/DASH inputs are therefore only reliable when the
    path is absolute. Remote URLs are passed through untouched.
    """
    if "://" in path:
        return path
    return os.path.abspath(path)


def _find_tool(name: str) -> Optional[str]:
    """Locate ffprobe/ffmpeg inside the active Python environment, else PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    env_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "Library", "bin", exe),
        os.path.join(env_root, "Library", "bin", exe),
        os.path.join(env_root, "bin", exe),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    from shutil import which

    return which(name)


FFPROBE = _find_tool("ffprobe")
FFMPEG = _find_tool("ffmpeg")


def _sane_frame_count(value) -> Optional[int]:
    """Reject the implausible frame counts these containers hand back."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count <= 0 or count > 10_000_000:
        return None
    return count


def _sane_fps(value) -> Optional[float]:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fps) or fps <= 0 or fps > 1000:
        return None
    return fps


def _sane_duration(value) -> Optional[float]:
    """Seconds. Unlike fps this has no sensible upper bound, so only the
    obviously-invalid ('N/A', negative, non-finite) is rejected."""
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(duration) or duration <= 0:
        return None
    return duration


@dataclass
class VideoProbe:
    """What the input actually is, as opposed to what its extension claims."""

    path: str
    container: Optional[str] = None
    codec: Optional[str] = None
    width: int = 0
    height: int = 0
    fps: Optional[float] = None
    duration: Optional[float] = None
    frame_count_estimate: Optional[int] = None
    pix_fmt: Optional[str] = None
    data_streams: List[str] = field(default_factory=list)
    has_klv: bool = False
    metadata_source: str = "opencv"
    decoder: str = "OpenCV (FFMPEG backend)"
    sequential_only: bool = False

    def describe(self) -> List[str]:
        """The lines reported to the user before inference starts."""
        fps_text = f"{self.fps:.3f}" if self.fps else "unknown"
        if self.duration:
            minutes, seconds = divmod(self.duration, 60)
            duration_text = f"{int(minutes)}m {seconds:04.1f}s ({self.duration:.2f}s)"
        else:
            duration_text = "unknown (not reported by this container)"
        frames_text = (f"~{self.frame_count_estimate} (estimate)"
                       if self.frame_count_estimate else "unknown -- decoded sequentially")
        lines = [
            f"Video: {os.path.basename(self.path)}",
            f"Container: {self.container or 'unknown'}",
            f"Codec: {self.codec or 'unknown'}"
            + (f" ({self.pix_fmt})" if self.pix_fmt else ""),
            f"Resolution: {self.width} x {self.height}",
            f"FPS: {fps_text}",
            f"Duration: {duration_text}",
            f"Frames: {frames_text}",
            f"Decoder: {self.decoder}",
            f"Metadata read by: {self.metadata_source}",
        ]
        if self.data_streams:
            lines.append(f"Non-video streams present: {', '.join(self.data_streams)}")
        if self.has_klv:
            lines.append(
                "KLV/MISB metadata stream detected -- the source file is read only, "
                "never rewritten. Any annotated video written by this tool is a "
                "derived product WITHOUT that metadata (see USER_GUIDE.md).")
        if self.sequential_only:
            lines.append(
                "Seeking is unreliable for this container; frames are decoded "
                "strictly sequentially.")
        return lines


def _probe_with_ffprobe(path: str) -> Optional[dict]:
    if not FFPROBE:
        return None
    cmd = [FFPROBE, "-v", "error", "-show_format", "-show_streams",
           "-of", "json", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _parse_rational(text) -> Optional[float]:
    """'30000/1001' -> 29.97002997. Exact rational beats OpenCV's rounded float."""
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        if "/" in str(text):
            num, den = str(text).split("/", 1)
            den_value = float(den)
            return float(num) / den_value if den_value else None
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe_video(path: str) -> VideoProbe:
    """Inspect the real container/stream, not just the file extension."""
    path = normalize_path(path)
    probe = VideoProbe(path=path)
    probe.sequential_only = os.path.splitext(path)[1].lower() in SEQUENTIAL_ONLY_EXTENSIONS

    data = _probe_with_ffprobe(path)
    if data:
        probe.metadata_source = "ffprobe"
        fmt = data.get("format", {})
        probe.container = fmt.get("format_long_name") or fmt.get("format_name")
        probe.duration = _sane_duration(fmt.get("duration"))
        video_stream = next((s for s in data.get("streams", [])
                             if s.get("codec_type") == "video"), None)
        if video_stream:
            probe.codec = video_stream.get("codec_long_name") or video_stream.get("codec_name")
            probe.width = int(video_stream.get("width") or 0)
            probe.height = int(video_stream.get("height") or 0)
            probe.pix_fmt = video_stream.get("pix_fmt")
            probe.fps = (_parse_rational(video_stream.get("avg_frame_rate"))
                         or _parse_rational(video_stream.get("r_frame_rate")))
            probe.frame_count_estimate = _sane_frame_count(video_stream.get("nb_frames"))
            if probe.duration is None:
                probe.duration = _sane_duration(video_stream.get("duration"))
        is_transport_stream = "mpegts" in (fmt.get("format_name") or "").lower()
        for stream in data.get("streams", []):
            if stream.get("codec_type") in ("data", "subtitle"):
                codec_name = (stream.get("codec_name") or stream.get("codec_type") or "").lower()
                tag = (stream.get("codec_tag_string") or "").lower()
                probe.data_streams.append(f"{stream.get('codec_type')}:{codec_name or tag or '?'}")
                explicit_klv = codec_name in _KLV_CODEC_NAMES or "klv" in tag
                generic_data_in_ts = is_transport_stream and codec_name in _GENERIC_DATA_CODEC_NAMES
                if explicit_klv or generic_data_in_ts:
                    probe.has_klv = True

    # OpenCV fills the gaps (and is the authority on whether WE can decode it).
    capture = cv2.VideoCapture(path)
    if capture.isOpened():
        probe.width = probe.width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        probe.height = probe.height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.fps = probe.fps or _sane_fps(capture.get(cv2.CAP_PROP_FPS))
        if probe.frame_count_estimate is None:
            probe.frame_count_estimate = _sane_frame_count(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        backend = capture.getBackendName()
        probe.decoder = f"OpenCV {cv2.__version__} ({backend} backend)"
        if probe.duration is None and probe.fps and probe.frame_count_estimate:
            probe.duration = probe.frame_count_estimate / probe.fps
        # Dimensions must describe the frames consumers actually receive, and
        # metadata does not: OpenCV applies rotation side-data (phone MP4/MOV
        # with rotate=90) at decode time, so a 640x360 stream comes out as
        # 360x640 frames while ffprobe AND CAP_PROP_FRAME_WIDTH/HEIGHT both
        # still say 640x360 (measured). The detector normalises boxes against
        # these numbers and the tracker uses the real frame shape, so a
        # mismatch silently misplaces every detection. One decoded frame is
        # the only trustworthy source; this handle is separate and released
        # immediately, so sequential reading elsewhere is unaffected.
        ok, frame = capture.read()
        if ok and frame is not None and frame.ndim >= 2:
            probe.height, probe.width = int(frame.shape[0]), int(frame.shape[1])
    capture.release()
    return probe


def validate_video(path: str) -> VideoProbe:
    """Probe the input and prove a frame can actually be decoded.

    Raises UnsupportedVideoError with an actionable message rather than
    letting SAM 3.1 or ArcGIS Pro fail later on an undecodable input.
    """
    path = normalize_path(path)
    if not os.path.isfile(path) and "://" not in path:
        raise UnsupportedVideoError(f"Video file not found: {path}")

    probe = probe_video(path)
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise UnsupportedVideoError(_cannot_open_message(path, probe))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise UnsupportedVideoError(
                f"'{os.path.basename(path)}' was opened but no frame could be decoded "
                f"(codec: {probe.codec or 'unknown'}). The file may be truncated, or its "
                f"codec may not be enabled in this environment's FFmpeg build.\n"
                + _transcode_hint(path))
    finally:
        capture.release()

    if not probe.width or not probe.height:
        probe.width = int(frame.shape[1])
        probe.height = int(frame.shape[0])
    return probe


def _cannot_open_message(path: str, probe: VideoProbe) -> str:
    ext = os.path.splitext(path)[1].lower()
    extra = ""
    if ext in (".mpd", ".m3u8"):
        extra = ("\nManifest inputs (.m3u8/.mpd) only open when every referenced "
                 "segment is reachable from the manifest's own location, and only "
                 "via an absolute path.")
    return (f"Could not open '{os.path.basename(path)}' with "
            f"{probe.decoder}. Container reported as: {probe.container or 'unknown'}."
            f"{extra}\n" + _transcode_hint(path))


def _transcode_hint(path: str) -> str:
    if not FFMPEG:
        return "Converting the file to H.264 MP4 with FFmpeg usually resolves this."
    base = os.path.splitext(os.path.basename(path))[0]
    return ("If this input is required, convert it first with the FFmpeg included in "
            "ArcGIS Pro:\n"
            f'  "{FFMPEG}" -i "{path}" -c:v h264_nvenc -pix_fmt yuv420p "{base}_converted.mp4"')


class VideoReader:
    """Sequential RGB frame reader with trustworthy timing.

    Forward-only by design: the pipeline reads frame t only after frame t-1,
    which matches how the tracker consumes frames and avoids seeking, which
    measurably does not work on MPEG-TS/M2TS or raw elementary streams here.
    """

    def __init__(self, path: str, probe: Optional[VideoProbe] = None, logger=None):
        self.path = normalize_path(path)
        self.log = logger
        self.probe = probe
        self.capture: Optional[cv2.VideoCapture] = None
        self.width = 0
        self.height = 0
        self.fps: Optional[float] = None
        self.frame_count_estimate: Optional[int] = None
        self.current_frame = -1          # index of the frame last returned
        self.current_timestamp = 0.0     # seconds
        self.container_timestamp: Optional[float] = None  # source PTS, None if unusable
        self._pts_usable = True
        self._last_pts = -1.0
        self._warned_pts = False

    # ------------------------------------------------------------------ open
    def open(self) -> VideoProbe:
        self.probe = self.probe or validate_video(self.path)
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise UnsupportedVideoError(_cannot_open_message(self.path, self.probe))

        self.width = self.probe.width or int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = self.probe.height or int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.probe.fps
        self.frame_count_estimate = self.probe.frame_count_estimate
        if self.fps is None and self.log:
            self.log.warning(
                "No usable frame rate for %s; timestamps will be frame indices.",
                os.path.basename(self.path))
        return self.probe

    # ------------------------------------------------------------------ read
    def read_frame(self) -> Optional[np.ndarray]:
        """Next frame as RGB uint8, or None at end of stream.

        End of stream is the ONLY loop terminator callers should use -- see the
        module docstring for why the reported frame count cannot be trusted.
        """
        if self.capture is None:
            raise RuntimeError("call open() first")
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            return None

        self.current_frame += 1
        self._update_timestamp()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _update_timestamp(self):
        """Prefer the container PTS, but only while it stays believable.

        CAP_PROP_POS_MSEC read after read() is the presentation timestamp of the
        frame just returned -- correct and non-integer-fps-safe where it works
        (verified: 33.37 ms steps on a 29.97 fps transport stream). On raw
        elementary streams it never leaves 0, so it is checked for forward
        progress and abandoned permanently the first time it misbehaves.
        """
        pts_seconds = None
        if self._pts_usable:
            raw = self.capture.get(cv2.CAP_PROP_POS_MSEC)
            if raw is not None and np.isfinite(raw) and raw >= 0:
                pts_seconds = float(raw) / 1000.0
                # Frame 0 legitimately sits at 0.0; after that it must advance.
                if self.current_frame > 0 and pts_seconds <= self._last_pts:
                    self._pts_usable = False
                    pts_seconds = None
            else:
                self._pts_usable = False

            if not self._pts_usable and not self._warned_pts:
                self._warned_pts = True
                if self.log:
                    self.log.warning(
                        "Container timestamps are not usable for %s; falling back to "
                        "frame-index timing.", os.path.basename(self.path))

        if pts_seconds is not None:
            self._last_pts = pts_seconds
            self.container_timestamp = pts_seconds
            self.current_timestamp = pts_seconds
        else:
            self.container_timestamp = None
            self.current_timestamp = (self.current_frame / self.fps) if self.fps \
                else float(self.current_frame)

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
