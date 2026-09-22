"""Streaming text-prompt detection, SAM tracking, and incremental output export."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Optional

from sam31_video_runtime import SAM31VideoRuntime, FrameReader
from detection_manager import DetectionManager
from track_manager import TrackManager
from track_state import TrackStateConfig
from text_prompts import parse_text_prompts
from video_reader import UnsupportedVideoError, normalize_path, validate_video
import arcgis_export as export


def validate_output_paths(video_path, *outputs):
    """Reject aliases before any writer can truncate the input or another output."""
    paths = [video_path] + [p for p in outputs if p]
    for i, path in enumerate(paths):
        for previous in paths[:i]:
            same = os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(previous))
            if os.path.exists(path) and os.path.exists(previous):
                same = same or os.path.samefile(path, previous)
            if same:
                raise ValueError(f"Source and output paths must be distinct: {previous!r}, {path!r}")


def _rss_mb() -> Optional[float]:
    """Resident set size of this process in MB, or None if psutil is absent."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def _gpu_mb():
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        return (torch.cuda.memory_allocated() / 2**20,
                torch.cuda.max_memory_allocated() / 2**20)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        return None, None


def log_memory(logger, label: str):
    """Emit a RAM/GPU datapoint. Memory must stay flat across a long video."""
    if logger is None:
        return
    rss = _rss_mb()
    gpu, gpu_peak = _gpu_mb()
    parts = [f"RAM {rss:.0f} MB" if rss is not None else "RAM n/a"]
    if gpu is not None:
        parts.append(f"GPU {gpu:.0f} MB (peak {gpu_peak:.0f} MB)")
    logger.info("memory [%s]: %s", label, ", ".join(parts))


def split_gdb_fc_path(fc_path: str):
    """Split a feature-class path into its geodatabase and dataset name.

    Feature-class output parameters in ArcGIS are a single path; the export
    helpers need the geodatabase and the feature-class name separately.
    """
    fc_path = os.path.normpath(fc_path)
    parts = fc_path.split(os.sep)
    for i, part in enumerate(parts):
        if part.lower().endswith(".gdb"):
            return os.sep.join(parts[:i + 1]), os.sep.join(parts[i + 1:])
    raise ValueError(f"expected a path inside a .gdb, got: {fc_path}")


def run_full_pipeline(video_path, text_prompt, detection_interval, confidence_threshold,
                       max_objects, runtime, iou_threshold: float = 0.30, centroid_threshold: float = 0.50,
                       preview_every: int = 60, save_annotated: bool = True, annotated_path: Optional[str] = None,
                       max_frames: Optional[int] = None,
                       progress_cb: Optional[Callable[[int, int], None]] = None,
                       memory_log_every: int = 250,
                       state_config: Optional[TrackStateConfig] = None,
                       bounded_history: bool = False, max_previews: int = 8,
                       stage_cb: Optional[Callable[[str], None]] = None, group_config=None):
    """Sequential decode, periodic grounding, persistent SAM tracking.

    progress_cb(frame_idx, n_frames), if given, is called once per frame --
    used by the geoprocessing tool to drive arcpy's progress bar and to check
    for a user-requested cancel.
    """
    prompts = parse_text_prompts(text_prompt)
    if detection_interval <= 0 or preview_every <= 0 or memory_log_every <= 0:
        raise ValueError("Detection, preview and memory logging intervals must be positive")
    if max_objects <= 0 or max_previews < 0:
        raise ValueError("max_objects must be positive and max_previews non-negative")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None")
    if save_annotated and not annotated_path:
        raise ValueError("An annotated output path is required when saving video")
    validate_output_paths(video_path, annotated_path if save_annotated else None)
    reader = FrameReader(video_path, logger=runtime.log)
    # Streaming detection: the detector is handed a 2-frame window as we decode
    # rather than the whole video up front (which allocated 50+ GiB and failed
    # on real FMV clips). See streaming_detector_state.
    try:
        runtime.open_video_streaming(video_path)
    except BaseException:
        reader.close()
        raise
    if runtime.session.is_active:
        runtime.session.close()                    # fresh tracking session; model weights untouched

    state_config = state_config or TrackStateConfig()
    tm = TrackManager(runtime, DetectionManager(iou_threshold, centroid_threshold),
                       max_objects=max_objects, logger=runtime.log,
                       source_video=reader.path, state_config=state_config,
                       bounded_history=bounded_history, group_config=group_config)

    writer = None
    previews = deque(maxlen=max_previews)
    # frame_count is an ESTIMATE and is wrong for several containers (negative
    # for raw elementary streams, short by one for MPEG-PS, long for truncated
    # transport streams), so it is only used for progress reporting. The decode
    # loop ends when the decoder reports end of stream -- using the estimate as
    # a bound silently processed zero frames on .h264/.h265 and dropped the
    # final frame on .mpg/.mpeg. See video_reader for the measurements.
    estimated_total = max_frames or reader.frame_count or 0

    # One-frame lookahead. A detection on frame N also reads frame N+1
    # (measured), so N+1 must already be decoded -- but it is simply the next
    # frame of the same strictly sequential read, so nothing is seeked or
    # decoded twice. `pending` holds at most one frame: memory stays O(1).
    pending = None
    peak_rss = _rss_mb() or 0.0
    log_memory(runtime.log, "before first frame")

    def read_next():
        """(frame, timestamp, container_timestamp) or None at end of stream."""
        nonlocal pending
        if pending is not None:
            item, pending = pending, None
            return item
        frame = reader.read()
        if frame is None:
            return None
        return frame, reader.current_timestamp, reader.container_timestamp

    def peek_next():
        """Buffer and return the following frame without consuming it."""
        nonlocal pending
        if pending is None:
            frame = reader.read()
            if frame is None:
                return None
            pending = (frame, reader.current_timestamp, reader.container_timestamp)
        return pending

    frame_idx = 0
    t_start = time.time()
    try:
        if save_annotated:
            writer = export.AnnotatedVideoWriter(annotated_path, reader.width, reader.height, reader.fps)
        while max_frames is None or frame_idx < max_frames:
            item = read_next()
            if item is None:
                break
            frame, timestamp, container_ts = item

            if frame_idx == 0 or frame_idx % detection_interval == 0:
                if stage_cb:
                    stage_cb(f"Searching for new objects: frame {frame_idx + 1}")
                # Install only what the detector will read: this frame and the
                # next one (absent at end of video -- the store clamps).
                window = [frame]
                numbers = [frame_idx]
                nxt = peek_next()
                if nxt is not None:
                    window.append(nxt[0])
                    numbers.append(frame_idx + 1)
                runtime.set_detection_window(window, numbers)
                detections, det_ms = [], 0.0
                for prompt in prompts:
                    if progress_cb:
                        progress_cb(frame_idx, estimated_total)
                    if stage_cb:
                        stage_cb(f"Searching for {prompt}: frame {frame_idx + 1}")
                    found, elapsed_ms = runtime.detect(frame_idx, prompt, prob_thresh=confidence_threshold)
                    detections.extend(found)
                    det_ms += elapsed_ms
                if stage_cb:
                    stage_cb(f"Tracking objects: frame {frame_idx + 1}")
                tm.bootstrap_or_redetect(frame, frame_idx, timestamp, detections, det_ms,
                                          source_timestamp=container_ts)
            else:
                tm.track_only(frame, frame_idx, timestamp,
                              source_timestamp=container_ts)

            if writer is not None or (max_previews and frame_idx % preview_every == 0):
                annotated = export.draw_annotations(
                    frame, tm.last_frame_rows(),
                    hide_box_when_lost=state_config.hide_box_when_lost,
                    show_lost_status=state_config.show_lost_status)
                if writer is not None:
                    writer.write(annotated)
                if max_previews and frame_idx % preview_every == 0:
                    previews.append((frame_idx, annotated))

            if progress_cb is not None:
                progress_cb(frame_idx, estimated_total)

            peak_rss = max(peak_rss, _rss_mb() or 0.0)
            if frame_idx and frame_idx % memory_log_every == 0:
                log_memory(runtime.log, f"frame {frame_idx}")

            frame_idx += 1
    except BaseException:
        tm.close_history()
        raise
    finally:
        if writer is not None:
            writer.close()
        reader.close()
        runtime.close_video()

    elapsed = time.time() - t_start
    runtime.log.info("processed %d frames in %.1fs (%.2f fps end-to-end)",
                      frame_idx, elapsed, frame_idx / elapsed if elapsed else 0.0)
    log_memory(runtime.log, "after last frame")
    runtime.log.info("peak RAM during processing: %.0f MB (bounded: independent of video length)",
                      peak_rss)
    tm.peak_rss_mb = round(peak_rss, 1)
    return tm, list(previews)


def run_pipeline_and_export(video_path, text_prompt, out_gdb=None, out_name=None, *,
                             detection_interval: int = 30, confidence_threshold: float = 0.35,
                             max_objects: int = 16, iou_threshold: float = 0.30, centroid_threshold: float = 0.50,
                             max_frames: Optional[int] = None,
                             save_annotated_video: bool = True, annotated_video_path: Optional[str] = None,
                             export_csv_flag: bool = True, out_csv_path: Optional[str] = None,
                             geometry: str = "polygon", logger=None,
                             progress_cb: Optional[Callable[[int, int], None]] = None,
                             runtime: Optional[SAM31VideoRuntime] = None,
                             warm_up_detector: bool = True,
                             memory_log_every: int = 250,
                             state_config: Optional[TrackStateConfig] = None,
                             return_results: bool = True, export_feature_class_flag: bool = True,
                             stage_cb: Optional[Callable[[str], None]] = None,
                             model_package_path: Optional[str] = None, group_config=None) -> dict:
    """Single entry point: load the model (unless an already-loaded `runtime` is
    passed in), run the pipeline once, export CSV + feature class, return a
    result dict. Owns and shuts down the runtime unless the caller supplied one.

    warm_up_detector: pay the one-time ~100x-slower-than-usual first detect()
    call (see SAM31VideoRuntime.warm_up_detector) up front, before the real
    pipeline loop, so it is reported as its own step rather than silently
    inflating frame 0's latency. Costs one extra video-open; safe to disable
    if the caller already warmed the detector up itself.

    return_results=False is the toolbox mode: disk-backed history, incremental
    exports, no retained preview pixels, and results=None. The default preserves
    the programmatic DataFrame return contract and its O(number of rows) memory.

    Export failures are returned in export_errors with success=False so callers
    can report independently completed files. Inference failures still raise.
    """
    # Validate and report the input BEFORE loading several GB of model weights,
    # so an undecodable video fails in seconds with a clear message instead of
    # after a long load or midway through inference.
    parse_text_prompts(text_prompt)  # reject empty lists before video/model initialization
    video_path = normalize_path(video_path)
    def stage(message):
        if logger:
            logger.info("%s", message)
        if stage_cb:
            stage_cb(message)

    if export_feature_class_flag and (not out_gdb or not out_name):
        raise ValueError("Choose a file-geodatabase feature-class path or disable feature-class export")
    if export_csv_flag and not out_csv_path:
        raise ValueError("A CSV output path is required when exporting CSV")
    if not any((export_csv_flag, export_feature_class_flag, save_annotated_video)):
        raise ValueError("Select at least one output: CSV, annotated video, or pixel-space feature class")
    if detection_interval <= 0 or max_objects <= 0 or memory_log_every <= 0:
        raise ValueError("Detection interval, max objects and memory logging interval must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None")
    if save_annotated_video and not annotated_video_path:
        raise ValueError("An annotated output path is required when saving video")
    validate_output_paths(video_path, annotated_video_path if save_annotated_video else None,
                          out_csv_path if export_csv_flag else None,
                          os.path.join(out_gdb, out_name) if export_feature_class_flag else None)
    stage("Validating video input...")
    probe = validate_video(video_path)
    if logger:
        for line in probe.describe():
            logger.info("%s", line)
    log_memory(logger, "before video")

    owns_runtime = runtime is None
    model_options = {}
    if model_package_path:
        if runtime is not None:
            raise ValueError('Pass either a model package or a supplied runtime, not both.')
        stage('Validating and preparing selected SAM 3.1 model package...')
        from model_package import prepare_package
        from package_model_cache import select_package
        assets = prepare_package(model_package_path, logger=logger)
        select_package(assets.fingerprint, logger)
        model_options = dict(checkpoint_path=assets.checkpoint, bpe_path=assets.vocabulary)
    if runtime is None:
        stage("Loading or reusing SAM 3.1 models...")
        runtime = SAM31VideoRuntime(confidence_threshold=confidence_threshold,
                                     detector_max_num_objects=max_objects, logger=logger, **model_options)
        runtime.load()
    log_memory(logger, "after tracker initialization")

    warm_up_ms = None
    tm = None
    try:
        if warm_up_detector:
            stage("Preparing detector (first run may take a few minutes)...")
            # Streaming warm-up: open the detector without preloading, then feed
            # it just the first two frames (what one detection reads). This is a
            # 2-frame read, not a replay of the video.
            runtime.open_video_streaming(video_path)
            warm_reader = FrameReader(video_path, logger=logger)
            try:
                warm_frames = []
                for _ in range(2):
                    frame = warm_reader.read()
                    if frame is None:
                        break
                    warm_frames.append(frame)
            finally:
                warm_reader.close()
            if warm_frames:
                runtime.set_detection_window(warm_frames, list(range(len(warm_frames))))
                warm_up_ms = runtime.warm_up_detector()

        stage("Processing video: tracking every frame and periodically searching for new objects...")
        tm, previews = run_full_pipeline(
            video_path, text_prompt, detection_interval, confidence_threshold, max_objects,
            runtime, iou_threshold=iou_threshold, centroid_threshold=centroid_threshold,
            save_annotated=save_annotated_video, annotated_path=annotated_video_path,
            max_frames=max_frames, progress_cb=progress_cb,
            memory_log_every=memory_log_every, state_config=state_config,
            bounded_history=not return_results, max_previews=8 if return_results else 0,
            stage_cb=stage_cb, group_config=group_config,
        )

        df = tm.results_dataframe() if return_results else tm.rows
        csv_path = fc_path = None
        export_errors = []
        for label, enabled, path, write in (
            ("CSV", export_csv_flag, out_csv_path, lambda: export.export_csv(df, out_csv_path)),
            ("Pixel-space feature class", export_feature_class_flag,
             os.path.join(out_gdb, out_name) if export_feature_class_flag else None,
             lambda: export.export_feature_class(df, out_gdb, out_name, geometry=geometry)),
        ):
            if not enabled:
                continue
            stage(f"Exporting {label}: {path}")
            try:
                written = write()
                if written is None:
                    raise RuntimeError("Exporter did not create an output")
                if label == "CSV":
                    csv_path = written
                else:
                    fc_path = written
            except Exception as exc:
                # Export failures must not discard independent completed products.
                message = f"{label} export failed at {path}: {exc}. This output may be incomplete."
                export_errors.append(message)
                if logger:
                    logger.warning("%s", message)

        performance = tm.performance_summary()
        performance["warm_up_ms"] = round(warm_up_ms, 0) if warm_up_ms is not None else None
        performance["peak_ram_mb"] = getattr(tm, "peak_rss_mb", None)
        return {
            "results": df if return_results else None,
            "csv_path": csv_path,
            "feature_class": fc_path,
            "annotated_video": annotated_video_path if save_annotated_video else None,
            "performance": performance,
            "previews": previews,
            "probe": probe,
            "export_errors": export_errors,
            "success": not export_errors,
        }
    finally:
        if tm is not None:
            tm.close_history()
        if owns_runtime:
            runtime.shutdown()
