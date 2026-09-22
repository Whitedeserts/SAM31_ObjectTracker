"""ArcGIS <-> SAM 3.1 box conversions.

ArcGIS ObjectTracker contract (Esri docs, "Use third-party object tracking
models with ArcGIS"):

    init_tracker(frame, boxes)   boxes rows: [object_id, x_min, y_min, x_max, y_max]  (pixels)
    track(frame) -> [[float(obj_id), float(x1), float(y1), float(x2), float(y2)], ...]
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
from torchvision.ops import masks_to_boxes

ArcGISBox = Tuple[int, float, float, float, float]


class BoxFormatError(ValueError):
    pass


MIN_BOX = 2.0        # minimum width/height (px) of any box handed to ArcGIS
INIT_MIN_BOX = 8.0   # boxes smaller than this at init are expanded around their centre


def sanitize_box(row, frame_w, frame_h, min_size=MIN_BOX):
    """Return `[float(id), x1, y1, x2, y2]` that is finite, inside the frame and at least
    `min_size` wide/high, or None if the row cannot be repaired. Never raises."""
    try:
        vals = [float(v) for v in row]
    except Exception:
        return None
    if len(vals) != 5 or not all(math.isfinite(v) for v in vals):
        return None
    oid, x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    W, H = float(frame_w), float(frame_h)
    x1, x2 = min(max(x1, 0.0), W), min(max(x2, 0.0), W)
    y1, y2 = min(max(y1, 0.0), H), min(max(y2, 0.0), H)
    if x2 - x1 < min_size:
        cx = (x1 + x2) / 2.0
        x1, x2 = cx - min_size / 2.0, cx + min_size / 2.0
        if x1 < 0:
            x1, x2 = 0.0, min_size
        if x2 > W:
            x1, x2 = max(0.0, W - min_size), W
    if y2 - y1 < min_size:
        cy = (y1 + y2) / 2.0
        y1, y2 = cy - min_size / 2.0, cy + min_size / 2.0
        if y1 < 0:
            y1, y2 = 0.0, min_size
        if y2 > H:
            y1, y2 = max(0.0, H - min_size), H
    if not (x2 > x1 and y2 > y1):
        return None
    return [float(int(round(oid))), x1, y1, x2, y2]


def parse_arcgis_boxes(boxes, frame_w, frame_h, min_size=INIT_MIN_BOX, lenient=True):
    """Normalise ArcGIS boxes into (id, x1, y1, x2, y2) tuples.

    Coordinates are clamped to the frame. Degenerate boxes (smaller than `min_size`
    after clamping) are expanded around their centre; non-finite rows and duplicate
    ids are dropped when `lenient` (the default), else raise BoxFormatError.
    A single box (5 numbers) is also accepted. Returns the list of boxes; the
    dropped rows are available in `parse_arcgis_boxes.last_warnings`.
    """
    warnings = []
    parse_arcgis_boxes.last_warnings = warnings
    if boxes is None:
        raise BoxFormatError("boxes is None")
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 5:
            raise BoxFormatError(f"expected [id, x1, y1, x2, y2], got {arr.tolist()}")
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise BoxFormatError(
            f"expected boxes of shape (N, 5) = [id, x_min, y_min, x_max, y_max], got {arr.shape}"
        )
    out: List[ArcGISBox] = []
    seen = set()
    for row in arr:
        if not np.isfinite(row).all():
            msg = f"non-finite box row {row.tolist()}"
            if not lenient:
                raise BoxFormatError(msg)
            warnings.append(msg)
            continue
        oid = int(round(row[0]))
        if oid in seen:
            msg = f"duplicate object id {oid} in init boxes (keeping the first)"
            if not lenient:
                raise BoxFormatError(msg)
            warnings.append(msg)
            continue
        fixed = sanitize_box(row, frame_w, frame_h, min_size=min_size)
        if fixed is None:
            msg = f"box for id {oid} cannot be placed inside {frame_w}x{frame_h}: {row.tolist()}"
            if not lenient:
                raise BoxFormatError(msg)
            warnings.append(msg)
            continue
        seen.add(oid)
        out.append((oid, fixed[1], fixed[2], fixed[3], fixed[4]))
    return out


def box_to_sam_corner_points(box: ArcGISBox, frame_w, frame_h):
    """ArcGIS pixel box -> (points (2,2) relative xy, labels (2,) int32).

    SAM 2 / SAM 3.1 prompt-encoder convention: label 2 = box top-left corner,
    label 3 = box bottom-right corner (see sam3/sam/prompt_encoder.py).
    """
    _, x1, y1, x2, y2 = box
    pts = torch.tensor(
        [[x1 / frame_w, y1 / frame_h], [x2 / frame_w, y2 / frame_h]], dtype=torch.float32
    )
    labels = torch.tensor([2, 3], dtype=torch.int32)
    return pts, labels


def _largest_component_box(crop: np.ndarray, join_nearby=False):
    """Bounding box (x1, y1, x2, y2; exclusive max) of the largest connected component of a
    2-D bool array, or None if empty. Falls back to the full extent if scipy is missing."""
    try:
        from scipy import ndimage
    except Exception:  # pragma: no cover
        ys, xs = np.nonzero(crop)
        return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if ys.size else None
    labels, n = ndimage.label(crop)
    if n == 0:
        return None
    if n == 1:
        ys, xs = np.nonzero(crop)
        return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    biggest = int(sizes.argmax())
    ys, xs = np.nonzero(labels == biggest)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    if join_nearby:
        # Only substantial neighbors of the largest component in this SAME
        # object mask are eligible. Do not chain fragments into distant regions.
        bx1, by1, bx2, by2 = box
        gap_limit = 0.15 * math.hypot(bx2 - bx1, by2 - by1)
        for label, slices in enumerate(ndimage.find_objects(labels), 1):
            if label == biggest or slices is None or sizes[label] < 0.15 * sizes[biggest]:
                continue
            sy, sx = slices
            dx = max(0, sx.start - bx2, bx1 - sx.stop)
            dy = max(0, sy.start - by2, by1 - sy.stop)
            if math.hypot(dx, dy) <= gap_limit:
                box = (min(box[0], sx.start), min(box[1], sy.start),
                       max(box[2], sx.stop), max(box[3], sy.stop))
    return box


def masks_to_arcgis_boxes(obj_ids, masks_bool: torch.Tensor, largest_component=True,
                          join_component_ids=()):
    """(N, H, W) bool masks -> list of [id, x1, y1, x2, y2] floats.

    Objects with an empty mask are skipped. x2/y2 are exclusive (inclusive
    pixel + 1) so the box covers the full pixel extent. With
    `largest_component=True` (default) the box is fitted to the largest
    connected component of the mask, so a few stray pixels far from the
    object (which SAM occasionally emits) cannot inflate the box. The
    component analysis runs on the tight crop of the mask only.
    """
    if masks_bool.numel() == 0:
        return []
    keep = masks_bool.flatten(1).any(dim=1)
    idx = torch.nonzero(keep, as_tuple=True)[0]
    if idx.numel() == 0:
        return []
    sel = masks_bool[idx]
    boxes = masks_to_boxes(sel.to(torch.uint8)).cpu().numpy()  # (K, 4) xyxy inclusive
    idx_np = idx.cpu().numpy()
    out = []
    crops_needed = []
    for k, i in enumerate(idx_np):
        x1, y1, x2, y2 = (int(v) for v in boxes[k])
        out.append([float(obj_ids[int(i)]), float(x1), float(y1), float(x2 + 1), float(y2 + 1)])
        if largest_component:
            crops_needed.append((k, x1, y1, x2 + 1, y2 + 1))
    if crops_needed:
        sel_cpu = sel.cpu().numpy()
        for k, x1, y1, x2, y2 in crops_needed:
            crop = sel_cpu[k, y1:y2, x1:x2]
            lc = _largest_component_box(crop, join_nearby=int(out[k][0]) in join_component_ids)
            if lc is not None:
                cx1, cy1, cx2, cy2 = lc
                out[k][1:5] = [float(x1 + cx1), float(y1 + cy1), float(x1 + cx2), float(y1 + cy2)]
    return out
