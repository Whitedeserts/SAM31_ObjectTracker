"""Export track tables, pixel-space feature classes and derived annotated video.

Feature geometry uses an unknown spatial reference. Sensor metadata is not
projected to ground coordinates. Derived video contains no KLV/MISB telemetry.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


def export_csv(df: pd.DataFrame, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def export_feature_class(df: pd.DataFrame, out_gdb: str, out_name: str,
                          geometry: str = "polygon") -> Optional[str]:
    """Write the tracks to a file geodatabase feature class in pixel space.

    geometry: "polygon" (the full box) or "point" (box centroid).
    Returns the feature-class path, or None if arcpy is unavailable.
    """
    try:
        import arcpy
    except ImportError:
        print("arcpy not available in this environment; skipping feature class export "
              "(the CSV export above is unaffected).")
        return None

    if not os.path.isdir(out_gdb):
        arcpy.management.CreateFileGDB(os.path.dirname(out_gdb), os.path.basename(out_gdb))

    fc_path = os.path.join(out_gdb, out_name)
    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    # Pixel space -- see module docstring. Omitting spatial_reference (rather than passing
    # an empty arcpy.SpatialReference()) is required: arcpy accepts the omission but rejects
    # an explicit "unknown" SpatialReference object with ERROR 000614 (confirmed on 3.7).
    geom_type = "POLYGON" if geometry == "polygon" else "POINT"
    arcpy.management.CreateFeatureclass(out_gdb, out_name, geom_type)
    fields = [
        ("frame_number", "LONG"), ("timestamp", "DOUBLE"),
        # source_timestamp is the container presentation timestamp, kept
        # alongside frame_number so a detection can be correlated back to the
        # source video's own KLV/MISB metadata at the same instant. Null when
        # the container does not supply usable timestamps (e.g. raw elementary
        # streams) -- see video_reader.
        ("source_timestamp", "DOUBLE"), ("source_video", "TEXT", 512),
        ("track_id", "LONG"),
        ("class_prompt", "TEXT", 64), ("confidence", "DOUBLE"),
        ("xmin", "DOUBLE"), ("ymin", "DOUBLE"), ("xmax", "DOUBLE"), ("ymax", "DOUBLE"),
        ("status", "TEXT", 16),
    ]
    grouped = 'member_track_ids' in df.columns
    if grouped:
        fields.append(('member_track_ids', 'TEXT', 128))
    for spec in fields:
        arcpy.management.AddField(fc_path, *spec)

    def _num(value):
        return None if value is None or pd.isna(value) else float(value)

    field_names = [f[0] for f in fields] + ["SHAPE@"]
    with arcpy.da.InsertCursor(fc_path, field_names) as cursor:
        for row in df.itertuples(index=False):
            x1, y1, x2, y2 = (_num(v) for v in (row.xmin, row.ymin, row.xmax, row.ymax))
            # Lost / terminated rows carry no position: null geometry, null
            # coordinates -- never the stale last box (see track_state).
            if None in (x1, y1, x2, y2):
                geom = None
            elif geometry == "polygon":
                # image row/col -> a y-up polygon (flip y so shapes look right-side-up if viewed)
                ring = [(x1, -y1), (x2, -y1), (x2, -y2), (x1, -y2), (x1, -y1)]
                geom = arcpy.Polygon(arcpy.Array([arcpy.Point(*p) for p in ring]))
            else:
                geom = arcpy.PointGeometry(arcpy.Point((x1 + x2) / 2.0, -(y1 + y2) / 2.0))
            source_ts = getattr(row, "source_timestamp", None)
            cursor.insertRow([
                int(row.frame_number), float(row.timestamp),
                _num(source_ts),
                str(getattr(row, "source_video", ""))[:512],
                int(row.track_id), str(row.class_prompt),
                _num(row.confidence), x1, y1, x2, y2, str(row.status),
            ] + ([str(row.member_track_ids)] if grouped else []) + [geom])
    print(f"wrote {len(df)} features to {fc_path}")
    return fc_path


# ---------------------------------------------------------------------------
# Visualisation
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
]


def _color_for(track_id: int):
    return _PALETTE[int(track_id) % len(_PALETTE)]


_VISIBLE = "VISIBLE"
_LOST_STATUSES = ("TEMPORARILY_LOST", "OUT_OF_FRAME")
MAX_STATUS_LINES = 8


def select_drawable_boxes(rows: pd.DataFrame, frame_w: int, frame_h: int, *,
                          hide_box_when_lost: bool = True) -> list:
    """Boxes that may be drawn this frame: VISIBLE rows with a sane box.

    Never returns a box for a lost or terminated track (their coordinates are
    null anyway), nor anything with NaN/Inf, zero area, or lying wholly
    outside the frame. Coordinates are clipped to the frame for drawing.
    Returns [(track_id, prompt, confidence, (x1, y1, x2, y2))].
    """
    from track_state import box_is_valid

    out = []
    for row in rows.itertuples(index=False):
        status = str(row.status)
        if status != _VISIBLE and (hide_box_when_lost or status not in _LOST_STATUSES):
            continue
        box = (row.xmin, row.ymin, row.xmax, row.ymax)
        if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in box):
            continue
        if not box_is_valid(box, frame_w, frame_h):
            continue
        x1 = int(round(max(0.0, float(row.xmin))))
        y1 = int(round(max(0.0, float(row.ymin))))
        x2 = int(round(min(float(frame_w - 1), float(row.xmax))))
        y2 = int(round(min(float(frame_h - 1), float(row.ymax))))
        if x2 <= x1 or y2 <= y1:
            continue
        conf = float(row.confidence) if row.confidence is not None and not pd.isna(row.confidence) else 0.0
        out.append((int(row.track_id), str(row.class_prompt), conf, (x1, y1, x2, y2)))
    return out


def lost_status_lines(rows: pd.DataFrame, max_lines: int = MAX_STATUS_LINES) -> list:
    """Compact text for tracks that are alive but not currently seen.

    e.g. ["Track 3 - lost", "Track 7 - out of frame", "+2 more lost"].
    """
    lines = []
    lost = [r for r in rows.itertuples(index=False) if str(r.status) in _LOST_STATUSES]
    lost.sort(key=lambda r: int(r.track_id))
    for row in lost[:max_lines]:
        kind = "out of frame" if str(row.status) == "OUT_OF_FRAME" else "lost"
        lines.append(f"Track {int(row.track_id)} - {kind}")
    if len(lost) > max_lines:
        lines.append(f"+{len(lost) - max_lines} more lost")
    return lines


def draw_annotations(frame_rgb: np.ndarray, rows: pd.DataFrame, *,
                     hide_box_when_lost: bool = True, show_lost_status: bool = True) -> np.ndarray:
    """rows: the results-table rows for exactly this frame. Returns a copy.

    Only VISIBLE tracks get a bounding box. A lost track is never shown as a
    frozen rectangle over the scene; instead it appears as a one-line entry in
    a small status panel in the top-left corner, so the viewer can tell the
    id is still alive without being misled about where the object is.
    """
    import cv2

    img = frame_rgb.copy()
    frame_h, frame_w = img.shape[:2]

    for track_id, prompt, conf, (x1, y1, x2, y2) in select_drawable_boxes(
            rows, frame_w, frame_h, hide_box_when_lost=hide_box_when_lost):
        color = _color_for(track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        label = f"id{track_id} {prompt} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    if show_lost_status:
        lines = lost_status_lines(rows)
        if lines:
            pad, line_h = 6, 18
            width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines) + 2 * pad
            height = len(lines) * line_h + 2 * pad
            panel = img[0:height, 0:width]
            cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.65, 0, panel)
            for i, text in enumerate(lines):
                cv2.putText(img, text, (pad, pad + (i + 1) * line_h - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


class AnnotatedVideoWriter:
    """Thin wrapper around cv2.VideoWriter for the optional annotated output video.

    This writes a NEW, DERIVED video: re-encoded MPEG-4 frames with boxes drawn
    on them. It carries no KLV/MISB metadata, no sensor/platform telemetry and
    no other data streams from the source, even when the source was an FMV
    transport stream that had them -- nothing here remultiplexes those streams.
    The output is therefore a visual review product and is NOT FMV-compliant;
    it must not be substituted for the original video in an FMV workflow. The
    source file itself is only ever opened for reading and is never rewritten.
    Use the source_video/source_timestamp columns in the tracking table to
    relate detections back to the original video and its metadata.
    """

    def __init__(self, out_path: str, width: int, height: int, fps: float):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps or 24.0, (width, height))
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Could not open annotated video output: {out_path}")
        self.out_path = out_path

    def write(self, frame_rgb: np.ndarray):
        import cv2

        self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()
