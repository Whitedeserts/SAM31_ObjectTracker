"""ArcGIS Pro Python toolbox wrapping the text-prompt SAM 3.1 tracking pipeline.
Add this toolbox in Pro via Catalog pane -> right-click Toolboxes -> Add
Toolbox, then browse to this .pyt file.
"""

import logging
import os
import sys
import re
from datetime import datetime

import arcpy

def _resolve_src_dir():
    toolbox_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(toolbox_dir, "src")
    if os.path.isfile(os.path.join(candidate, "pipeline.py")):
        return candidate
    raise RuntimeError('Release files are missing. Extract the complete release ZIP and add its toolbox again.')


# Application modules refreshed before each run. detector_cache and
# sam31_runtime.model_loader deliberately survive so model weights are reused.
_TOOLBOX_SRC_MODULES = (
    "pipeline", "sam31_video_runtime", "streaming_detector_state", "track_manager",
    "detection_manager", "arcgis_export", "video_reader", "track_state", "result_store",
    "sam31_runtime.sam31_session",
    "streaming_session_pool", "track_quality", "logical_groups",
    "model_package", "text_prompts",
)


def _import_pipeline():
    """Restore the release import path at execution time.

    ArcGIS Pro may restore sys.path after toolbox loading. Refresh application
    modules while preserving process-wide model caches and pinned SAM classes.
    """
    import importlib

    src_dir = _resolve_src_dir()
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    importlib.invalidate_caches()

    # Refresh lightweight modules; cached models retain their original SAM classes.
    for name in _TOOLBOX_SRC_MODULES:
        sys.modules.pop(name, None)
    try:
        import pipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not import src/pipeline.py from {src_dir} "
            f"(exists={os.path.isfile(os.path.join(src_dir, 'pipeline.py'))}). "
            f"Missing dependency: {exc.name}. Use the supported ArcGIS Pro deep-learning environment; see USER_GUIDE.md.") from exc
    return pipeline


def _supported_extensions():
    """Extension filter for the input parameter, from the single source of truth.

    getParameterInfo runs at toolbox-load time, when sys.path may not carry
    src/ yet, so this falls back to a literal list rather than letting the
    whole toolbox fail to load over a browse filter.
    """
    try:
        src_dir = _resolve_src_dir()
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from video_reader import SUPPORTED_EXTENSIONS

        return list(SUPPORTED_EXTENSIONS)
    except Exception:  # noqa: BLE001 - a browse filter must never block loading
        return ["mp4", "mov", "avi", "mkv", "ts", "m2ts", "mpg", "mpeg", "ps",
                "vob", "wmv", "h264", "h265", "m3u8", "mpd"]


SRC_DIR = _resolve_src_dir()
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def value_or_default(parameter, default):
    """Zero is an explicit setting, particularly for loss grace periods."""
    return default if parameter.value in (None, "") else parameter.value


class Toolbox:
    def __init__(self):
        self.label = "SAM 3.1 Text-Prompt Tracking"
        self.alias = "sam31TextPromptTracking"
        self.tools = [SAM31TextPromptTrackingTool]


class _ArcpyLogHandler(logging.Handler):
    """Forwards the pipeline's own logger straight into the GP messages pane."""

    def emit(self, record):
        msg = self.format(record)
        if record.levelno >= logging.ERROR:
            arcpy.AddError(msg)
        elif record.levelno >= logging.WARNING:
            arcpy.AddWarning(msg)
        else:
            arcpy.AddMessage(msg)


class SAM31TextPromptTrackingTool:
    def __init__(self):
        self.label = "Track Objects by Text Prompt (SAM 3.1)"
        self.description = (
            "Detects objects matching a text prompt (e.g. 'truck', 'person') in a "
            "video using SAM 3.1, initializes tracks for them, tracks continuously "
            "with SAM 3.1 Object Multiplex, and periodically re-runs detection to "
            "discover new objects without duplicating existing tracks.\n\n"
            "The simultaneous-object limit defaults to 16 and can be increased. "
            "Higher limits use more GPU memory and processing time. Reduce the "
            "limit if GPU memory runs out or processing is too slow.\n\n"
            "Validated input formats: MP4, MOV, AVI, MKV, MPEG-TS (.ts), M2TS, "
            "MPG/MPEG, PS, VOB, WMV, raw H.264/H.265 elementary streams, and local "
            "HLS (.m3u8) / DASH (.mpd) manifests. Inputs are identified by actual "
            "container and codec, not by file extension, and are validated before "
            "processing starts.\n\n"
            "FMV note: SAM 3.1 reads video frames only. The source file is never "
            "modified, but any annotated video produced here is a derived visual "
            "product that does NOT carry KLV/MISB metadata and is not FMV-compliant. "
            "Use the source_video and source_timestamp output fields to relate "
            "detections back to the original video's metadata. "
            "See USER_GUIDE.md for formats and validation limits."
        )
        self.canRunInBackground = False
        self._generated_paths = {}

    # ------------------------------------------------------------------ params
    def getParameterInfo(self):
        model_package = arcpy.Parameter(
            displayName='SAM 3.1 Model Package (.dlpk)', name='model_package',
            datatype='DEFile', parameterType='Required', direction='Input')
        model_package.filter.list = ['dlpk']
        model_package.description = 'Local compatible SAM 3.1 DLPK. Verified assets are cached per user; no upload or package code execution.'
        in_video = arcpy.Parameter(
            displayName="Input Video", name="in_video", datatype="DEFile",
            parameterType="Required", direction="Input")
        # Only extensions validated end-to-end through this toolbox's decoder
        # are advertised here -- see USER_GUIDE.md. The list comes
        # from video_reader so the filter and the docs cannot drift apart.
        in_video.filter.list = _supported_extensions()
        in_video.description = (
            "Local video file. MPEG-TS (.ts) and M2TS FMV captures are supported and "
            "are read sequentially; the file is opened read-only and never rewritten. "
            "HLS/DASH manifests require their segment files to be present locally.")

        text_prompt = arcpy.Parameter(
            displayName="Text Prompts (separate with commas or semicolons)",
            name="text_prompt", datatype="GPString", parameterType="Required", direction="Input")

        out_feature_class = arcpy.Parameter(
            displayName="Pixel-Space Feature Class (Not Georeferenced)", name="out_feature_class", datatype="DEFeatureClass",
            parameterType="Optional", direction="Output", category="Output Paths (optional overrides)")
        out_feature_class.enabled = False

        output_folder = arcpy.Parameter(displayName="Output Folder", name="output_folder",
            datatype="DEFolder", parameterType="Required", direction="Input")
        run_name = arcpy.Parameter(displayName="Run Name", name="run_name",
            datatype="GPString", parameterType="Required", direction="Input")
        run_mode = arcpy.Parameter(displayName="Process", name="run_mode",
            datatype="GPString", parameterType="Required", direction="Input")
        run_mode.filter.list = ["Full video", "Short trial"]
        run_mode.value = "Full video"
        export_features = arcpy.Parameter(displayName="Export Pixel-Space Feature Class (Not Georeferenced)",
            name="export_features", datatype="GPBoolean", parameterType="Optional", direction="Input",
            category="Outputs")
        export_features.value = False

        detection_interval = arcpy.Parameter(
            displayName="Detection Interval (frames between re-detection passes)",
            name="detection_interval", datatype="GPLong", parameterType="Optional", direction="Input",
            category="Detection Settings")
        detection_interval.value = 30

        confidence_threshold = arcpy.Parameter(
            displayName="Confidence Threshold", name="confidence_threshold", datatype="GPDouble",
            parameterType="Optional", direction="Input", category="Detection Settings")
        confidence_threshold.value = 0.35

        max_objects = arcpy.Parameter(
            displayName="Max Simultaneous Objects", name="max_objects", datatype="GPLong",
            parameterType="Optional", direction="Input", category="Detection Settings")
        max_objects.value = 16

        iou_threshold = arcpy.Parameter(
            displayName="Duplicate-Match IoU Threshold", name="iou_threshold", datatype="GPDouble",
            parameterType="Optional", direction="Input", category="Advanced Matching")
        iou_threshold.value = 0.30

        centroid_threshold = arcpy.Parameter(
            displayName="Duplicate-Match Centroid Distance Threshold (normalized)",
            name="centroid_threshold", datatype="GPDouble", parameterType="Optional", direction="Input",
            category="Advanced Matching")
        centroid_threshold.value = 0.50

        max_frames = arcpy.Parameter(
            displayName="Trial Frames (first frames of video)", name="max_frames",
            datatype="GPLong", parameterType="Optional", direction="Input")
        max_frames.value = 90
        max_frames.enabled = False

        lost_grace_frames = arcpy.Parameter(
            displayName="Lost Grace Frames (occluded: frames to wait before terminating)",
            name="lost_grace_frames", datatype="GPLong", parameterType="Optional",
            direction="Input", category="Occlusion Handling")
        lost_grace_frames.value = 30

        out_of_frame_grace_frames = arcpy.Parameter(
            displayName="Out-of-Frame Grace Frames (left the frame: shorter wait)",
            name="out_of_frame_grace_frames", datatype="GPLong", parameterType="Optional",
            direction="Input", category="Occlusion Handling")
        out_of_frame_grace_frames.value = 10

        min_valid_mask_area = arcpy.Parameter(
            displayName="Min Valid Mask Area (pixels)", name="min_valid_mask_area",
            datatype="GPLong", parameterType="Optional", direction="Input",
            category="Occlusion Handling")
        min_valid_mask_area.value = 64

        min_track_confidence = arcpy.Parameter(
            displayName="Min Track Confidence (blank = tracker threshold)",
            name="min_track_confidence", datatype="GPDouble", parameterType="Optional",
            direction="Input", category="Occlusion Handling")

        hide_box_when_lost = arcpy.Parameter(
            displayName="Hide Box When Lost", name="hide_box_when_lost", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category="Occlusion Handling")
        hide_box_when_lost.value = True

        show_lost_status = arcpy.Parameter(
            displayName="Show Lost-Track Status Panel", name="show_lost_status",
            datatype="GPBoolean", parameterType="Optional", direction="Input",
            category="Occlusion Handling")
        show_lost_status.value = True

        save_annotated_video = arcpy.Parameter(
            displayName="Save Annotated Video", name="save_annotated_video", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category="Output Options")
        save_annotated_video.value = True

        annotated_video_path = arcpy.Parameter(
            displayName="Annotated Video Output", name="annotated_video_path", datatype="DEFile",
            parameterType="Optional", direction="Output", category="Output Options")

        export_csv = arcpy.Parameter(
            displayName="Export Results CSV", name="export_csv", datatype="GPBoolean",
            parameterType="Optional", direction="Input", category="Output Options")
        export_csv.value = True

        csv_path = arcpy.Parameter(
            displayName="CSV Output", name="csv_path", datatype="DEFile",
            parameterType="Optional", direction="Output", category="Output Options")

        detection_interval.description = "How often to search for new objects. Larger values reduce detection work but delay discovery. Existing objects track every frame."
        max_objects.description = "Maximum objects tracked at the same time, not the total IDs created across the video. Default: 16. Higher values use more GPU memory and may slow processing. Reduce this value if GPU memory runs out or processing is too slow. Lost tracks within their grace period and grouped member tracks still count toward this limit."
        lost_grace_frames.description = "Number of missing frames to retain an occluded object's ID before ending its track."
        out_of_frame_grace_frames.description = "Number of missing frames to retain a track last seen at an image edge."
        save_annotated_video.description = "Visual review copy without KLV/MISB metadata. Disabling saves drawing and encoding time."
        out_feature_class.description = "Optional bounding-box polygons in video pixel coordinates, not geographic map locations."
        max_frames.description = "A short trial processes the first 90 frames by default. Initial model preparation still applies."
        text_prompt.description = "Enter one or more object categories, separated by commas or semicolons, for example car; swimming pool or car, swimming pool. Each category is searched separately using the same model. More categories increase detection time. All categories share the simultaneous-object limit. Each track keeps its initial category label."
        for parameter in (save_annotated_video, export_csv):
            parameter.category = "Outputs"
        for parameter in (csv_path, annotated_video_path):
            parameter.category = "Output Paths (optional overrides)"
        group_vehicle = arcpy.Parameter(displayName="Group Object Parts",
            name="group_vehicle", datatype="GPBoolean", parameterType="Optional",
            direction="Input", category="Advanced Matching")
        group_vehicle.value = False
        group_vehicle.description = "May combine tracks for parts of the same object into one output box and ID when their shape and movement provide consistent evidence. Best suited to connected, elongated parts, such as a truck and trailer. Can reduce extra boxes, but incorrect grouping is possible. May increase processing time, especially with many objects. Leave off for separate people or animals."
        group_confirm = arcpy.Parameter(displayName="Grouping Confirmation Time (seconds)",
            name="group_confirm", datatype="GPDouble", parameterType="Optional",
            direction="Input", category="Advanced Matching")
        group_confirm.value = .5
        group_confirm.enabled = False
        group_confirm.description = "How long supporting evidence must persist before tracks are grouped. Larger values delay grouping and reduce brief, accidental matches."
        group_release = arcpy.Parameter(displayName="Grouping Separation Time (seconds)",
            name="group_release", datatype="GPDouble", parameterType="Optional",
            direction="Input", category="Advanced Matching")
        group_release.value = 1.
        group_release.enabled = False
        group_release.description = "How long separation evidence must persist before grouped tracks become independent again. Temporary missing observations do not count as separation."
        return [in_video, text_prompt, output_folder, model_package, run_name, run_mode, max_frames,
                export_csv, save_annotated_video, export_features, csv_path, annotated_video_path,
                out_feature_class, detection_interval, confidence_threshold,
                max_objects, iou_threshold, centroid_threshold,
                lost_grace_frames, out_of_frame_grace_frames, min_valid_mask_area,
                min_track_confidence, hide_box_when_lost, show_lost_status,
                group_vehicle, group_confirm, group_release]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        p = {par.name: par for par in parameters}

        for key in ('group_confirm', 'group_release'):
            p[key].enabled = bool(p['group_vehicle'].value)

        if not p["run_name"].valueAsText and p["in_video"].valueAsText:
            stem = os.path.splitext(os.path.basename(p["in_video"].valueAsText))[0]
            stem = re.sub(r"[^A-Za-z0-9_]", "_", stem)[:60]
            p["run_name"].value = "Tracks_" + stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        p["max_frames"].enabled = p["run_mode"].valueAsText == "Short trial"
        folder, name = p["output_folder"].valueAsText, p["run_name"].valueAsText
        defaults = {}
        if folder and name:
            defaults = {
                "csv_path": os.path.join(folder, name + ".csv"),
                "annotated_video_path": os.path.join(folder, name + "_annotated.mp4"),
                "out_feature_class": os.path.join(folder, "SAM31_Tracks.gdb", name + "_pixels"),
            }
        for key, toggle in (("csv_path", "export_csv"), ("annotated_video_path", "save_annotated_video"),
                            ("out_feature_class", "export_features")):
            parameter = p[key]
            parameter.enabled = bool(p[toggle].value)
            if key in defaults and (not parameter.valueAsText or
                    parameter.valueAsText == self._generated_paths.get(key)):
                parameter.value = defaults[key]
                self._generated_paths[key] = defaults[key]

    def updateMessages(self, parameters):
        p = {par.name: par for par in parameters}
        for parameter in parameters:
            parameter.clearMessage()
        if p['group_vehicle'].value:
            import math
            for key in ('group_confirm', 'group_release'):
                value = float(value_or_default(p[key], .5 if key == 'group_confirm' else 1.))
                if not math.isfinite(value) or value <= 0:
                    p[key].setErrorMessage('Enter a finite positive duration in seconds.')
        for key in ("model_package", "in_video", "text_prompt", "output_folder", "run_name"):
            if not (p[key].valueAsText or "").strip():
                p[key].setErrorMessage("This value is required.")
        src_dir = _resolve_src_dir()
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from text_prompts import parse_text_prompts
        try:
            parse_text_prompts(p['text_prompt'].valueAsText)
        except ValueError as exc:
            p['text_prompt'].setErrorMessage(str(exc))
        if p['model_package'].valueAsText:
            try:
                # Pro can restore sys.path between toolbox load and validation.
                src_dir = _resolve_src_dir()
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                from model_package import inspect_package
                inspect_package(p['model_package'].valueAsText)
            except ValueError as exc:
                p['model_package'].setErrorMessage(str(exc))
        folder = p["output_folder"].valueAsText
        if folder and not os.path.isdir(folder):
            p["output_folder"].setErrorMessage("Choose an existing output folder.")
        if not any(bool(p[key].value) for key in ("export_csv", "save_annotated_video", "export_features")):
            p["export_csv"].setErrorMessage("Select at least one output.")
        name = p["run_name"].valueAsText
        if name and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,99}", name):
            p["run_name"].setErrorMessage("Use 1-100 letters, numbers or underscores; start with a letter.")
        for key in ("detection_interval", "max_objects"):
            if p[key].value is not None and int(p[key].value) < 1:
                p[key].setErrorMessage("Enter a positive whole number.")
        if p["max_objects"].value and int(p["max_objects"].value) > 16:
            p["max_objects"].setWarningMessage("Higher object limits use more GPU memory and processing time. Reduce this value if GPU memory runs out or processing is too slow.")
        if p["run_mode"].valueAsText == "Short trial" and int(value_or_default(p["max_frames"], 90)) < 1:
            p["max_frames"].setErrorMessage("Trial frames must be positive.")
        for key in ("lost_grace_frames", "out_of_frame_grace_frames", "min_valid_mask_area"):
            if p[key].value is not None and int(p[key].value) < 0:
                p[key].setErrorMessage("Enter zero or a positive whole number.")
        for key in ("confidence_threshold", "min_track_confidence", "iou_threshold"):
            if p[key].value is not None and not 0 <= float(p[key].value) <= 1:
                p[key].setErrorMessage("Enter a value between 0 and 1.")
        if p["export_features"].value and p["out_feature_class"].valueAsText:
            if ".gdb" + os.sep not in os.path.normpath(p["out_feature_class"].valueAsText).lower():
                p["out_feature_class"].setErrorMessage("Choose a feature class inside a file geodatabase (.gdb).")
        for key, toggle in (("csv_path", "export_csv"), ("annotated_video_path", "save_annotated_video"),
                            ("out_feature_class", "export_features")):
            if p[toggle].value and not p[key].valueAsText:
                p[key].setErrorMessage("Choose an output folder and run name, or enter an output path.")

    # ------------------------------------------------------------------ run
    def execute(self, parameters, messages):
        pipeline = _import_pipeline()
        self.updateParameters(parameters)
        p = {par.name: par for par in parameters}
        self.updateMessages(parameters)
        if any(par.hasError() for par in parameters):
            raise arcpy.ExecuteError("Correct the highlighted parameters before running.")
        video_path = p["in_video"].valueAsText
        text_prompt = p["text_prompt"].valueAsText
        export_features = bool(p["export_features"].value)
        out_gdb, out_name = pipeline.split_gdb_fc_path(p["out_feature_class"].valueAsText) if export_features else (None, None)

        detection_interval = int(value_or_default(p["detection_interval"], 30))
        confidence_threshold = float(value_or_default(p["confidence_threshold"], 0.35))
        max_objects = int(value_or_default(p["max_objects"], 16))
        iou_threshold = float(value_or_default(p["iou_threshold"], 0.30))
        centroid_threshold = float(value_or_default(p["centroid_threshold"], 0.50))
        max_frames = int(value_or_default(p["max_frames"], 90)) if p["run_mode"].valueAsText == "Short trial" else None
        save_annotated_video = bool(p["save_annotated_video"].value)
        annotated_video_path = p["annotated_video_path"].valueAsText if save_annotated_video else None

        from track_state import TrackStateConfig

        min_conf_value = p["min_track_confidence"].value
        state_config = TrackStateConfig(
            lost_grace_frames=int(value_or_default(p["lost_grace_frames"], 30)),
            out_of_frame_grace_frames=int(value_or_default(p["out_of_frame_grace_frames"], 10)),
            min_valid_mask_area=int(value_or_default(p["min_valid_mask_area"], 64)),
            min_track_confidence=(float(min_conf_value) if min_conf_value not in (None, "") else None),
            hide_box_when_lost=bool(p["hide_box_when_lost"].value),
            show_lost_status=bool(p["show_lost_status"].value),
        )
        export_csv_flag = bool(p["export_csv"].value)
        csv_path = p["csv_path"].valueAsText if export_csv_flag else None

        logger = logging.getLogger("sam31_pipeline_tool")
        logger.setLevel(logging.INFO)
        logger.handlers[:] = []
        handler = _ArcpyLogHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

        arcpy.SetProgressor(
            "default",
            "Loading SAM 3.1 models and warming up the detector "
            "(one-time per Pro session, can take a few minutes)...")

        def progress_cb(frame_idx, n_frames):
            if frame_idx % 15 == 0:
                total = f" / approximately {n_frames}" if n_frames else " (total unknown)"
                arcpy.SetProgressorLabel(f"Tracking: {frame_idx + 1} frames processed{total}")
            try:
                if arcpy.env.isCancelled:
                    raise arcpy.ExecuteError("Cancelled by user.")
            except AttributeError:
                pass

        # Validate the input before anything expensive happens, and turn an
        # undecodable video into a clean tool error rather than a traceback.
        from video_reader import UnsupportedVideoError, validate_video

        arcpy.SetProgressorLabel("Validating video input...")
        try:
            probe = validate_video(video_path)
        except UnsupportedVideoError as exc:
            for line in str(exc).splitlines():
                arcpy.AddError(line)
            raise arcpy.ExecuteError(str(exc))

        for line in probe.describe():
            arcpy.AddMessage(line)
        if save_annotated_video and probe.has_klv:
            arcpy.AddWarning(
                "Source carries a KLV/MISB metadata stream. The annotated video is a "
                "derived visual product and will NOT contain that metadata or be "
                "FMV-compliant; the source file is not modified. Use the "
                "source_video/source_timestamp fields to relate detections to the "
                "original FMV metadata.")

        from logical_groups import GroupConfig
        result = pipeline.run_pipeline_and_export(
            video_path, text_prompt, out_gdb, out_name,
            detection_interval=detection_interval, confidence_threshold=confidence_threshold,
            max_objects=max_objects, iou_threshold=iou_threshold, centroid_threshold=centroid_threshold,
            max_frames=max_frames, save_annotated_video=save_annotated_video,
            annotated_video_path=annotated_video_path, export_csv_flag=export_csv_flag,
            out_csv_path=csv_path, logger=logger, progress_cb=progress_cb,
            state_config=state_config, return_results=False,
            export_feature_class_flag=export_features, stage_cb=arcpy.SetProgressorLabel,
            model_package_path=p['model_package'].valueAsText,
            group_config=(GroupConfig(enabled=True,
                confirm_seconds=float(value_or_default(p['group_confirm'], .5)),
                release_seconds=float(value_or_default(p['group_release'], 1.)))
                if p['group_vehicle'].value else GroupConfig()),
        )

        if result['feature_class']:
            arcpy.AddMessage(f"Wrote pixel-space feature class (NOT georeferenced): {result['feature_class']}")
        if result["csv_path"]:
            arcpy.AddMessage(f"Wrote CSV: {result['csv_path']}")
        if result["annotated_video"]:
            arcpy.AddMessage(f"Wrote annotated video: {result['annotated_video']}")
        perf = result["performance"]
        arcpy.AddMessage(f"Processed {perf.get('frames_processed', 0)} frames; created {perf.get('total_tracks_created', 0)} tracks.")
        if not perf.get('total_tracks_created', 0):
            arcpy.AddWarning("No objects were tracked. Try a clearer prompt or review the confidence threshold with a short trial.")
        if perf.get("warm_up_ms") is not None:
            arcpy.AddMessage(f"Detector warm-up (one-time): {perf['warm_up_ms']:.0f} ms")
        arcpy.AddMessage(
            "Performance: {frames_processed} frames, {detection_passes} detection pass(es), "
            "avg track {avg_track_latency_ms} ms/frame, avg detect {avg_detect_latency_ms} ms/pass, "
            "~{approx_fps} fps, {total_tracks_created} track(s) created, "
            "GPU peak {gpu_peak_mb} MB".format(**perf))
        arcpy.AddMessage(
            "Track lifecycle: {tracks_terminated} track(s) terminated after their grace period, "
            "{same_id_recoveries} same-ID recover(ies) after temporary loss".format(
                tracks_terminated=perf.get("tracks_terminated", 0),
                same_id_recoveries=perf.get("same_id_recoveries", 0)))

        for key, result_key in (("out_feature_class", "feature_class"), ("csv_path", "csv_path"),
                                ("annotated_video_path", "annotated_video")):
            p[key].value = result[result_key]
        arcpy.ResetProgressor()
        if result.get("export_errors"):
            for error in result["export_errors"]:
                arcpy.AddError(error)
            raise arcpy.ExecuteError("Video processing completed, but an output export failed. Successfully written outputs are listed above.")
        arcpy.AddMessage("Short trial completed successfully." if max_frames else "Video processing and selected exports completed successfully.")

    def postExecute(self, parameters):
        return

