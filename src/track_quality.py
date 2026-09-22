"""Bounded mask descriptors and conservative observation geometry checks."""
import math

import numpy as np


def compact_mask(mask, max_side=384):
    """A current-frame descriptor only; never retain full-resolution mask history."""
    import cv2
    mask = np.asarray(mask, dtype=np.uint8)
    h, w = mask.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        mask = cv2.resize(mask.astype(np.float32),
                          (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    return mask > 0


def mask_overlap(a, b):
    """Return IoU and intersection divided by each mask's area."""
    if a is None or b is None or a.shape != b.shape:
        return None
    aa, bb = int(a.sum()), int(b.sum())
    if not aa or not bb:
        return (0.0, 0.0, 0.0)
    inter = int(np.count_nonzero(a & b))
    return inter / (aa + bb - inter), inter / aa, inter / bb


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def mask_box(mask):
    """Exclusive pixel bounds of the final mask, without materializing all pixels."""
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if not len(xs) or not len(ys):
        return None
    return [float(xs[0]), float(ys[0]), float(xs[-1] + 1), float(ys[-1] + 1)]


def plausible_geometry(previous, box, elapsed=1, reference=None):
    """Reject abrupt expansion/teleportation, not absolute object size or zoom.

    Recovery permits bounded additional displacement with elapsed time. Area
    remains anchored to the last trusted observation so repeated bad masks
    cannot gradually validate themselves.
    """
    if previous is None:
        return True
    reference = reference or previous
    old, new = area(reference), area(box)
    if min(old, new) <= 0:
        return False
    ratio = new / old
    # Partial occlusion may shrink a legitimate object abruptly. Expansion is
    # compared with its recent full extent, never just the final visible sliver.
    if ratio > 3:
        return False
    dx = (box[0] + box[2] - previous[0] - previous[2]) / 2
    dy = (box[1] + box[3] - previous[1] - previous[3]) / 2
    diagonal = max(8, math.hypot(reference[2] - reference[0], reference[3] - reference[1]))
    return math.hypot(dx, dy) <= diagonal * min(2.0, 0.5 + 0.1 * max(1, elapsed))
