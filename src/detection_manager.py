"""Associate periodic detections and reject redundant object hypotheses.

Primary association is one-to-one using mask support and conservative box
fallbacks. A separate rejection pass checks remaining candidates against
already matched tracks and candidates from the same detector call. Nearness
alone never merges two vehicles. Last trusted masks remain bounded by the
live-track limit; they do not replace SAM temporal memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from sam31_video_runtime import Detection
from track_quality import compact_mask, mask_overlap


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid_dist_norm(a: Sequence[float], b: Sequence[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    dist = math.hypot(acx - bcx, acy - bcy)
    diag_a = math.hypot(a[2] - a[0], a[3] - a[1])
    diag_b = math.hypot(b[2] - b[0], b[3] - b[1])
    scale = max(1e-6, (diag_a + diag_b) / 2.0)
    return dist / scale


@dataclass
class MatchResult:
    new_detections: List[Detection]                 # genuinely new objects
    matched: Dict[int, Detection]                    # track_id -> the detection that confirmed it
    matched_pairs: List[Tuple[int, int, float]]       # (track_id, detection_index, iou) for logging
    suppressed: int = 0


class DetectionManager:
    def __init__(self, iou_threshold: float = 0.30, centroid_threshold: float = 0.50):
        self.iou_threshold = iou_threshold
        self.centroid_threshold = centroid_threshold

    def is_same_object(self, det_box, track_box) -> Tuple[bool, float]:
        iou = _iou(det_box, track_box)
        if iou >= self.iou_threshold:
            return True, iou
        if iou > 0.05 and _centroid_dist_norm(det_box, track_box) <= min(self.centroid_threshold, 0.25):
            return True, iou
        return False, iou

    def match(self, detections: List[Detection], active_boxes: Dict[int, List[float]],
              active_masks=None, active_prompts=None) -> MatchResult:
        """active_boxes: track_id -> [x1, y1, x2, y2] (use the tracker's held boxes,
        which are valid even for currently-occluded/lost tracks -- see module docstring)."""
        active_masks = active_masks or {}
        active_prompts = active_prompts or {}
        def different(prompt, other):
            return other is not None and prompt.strip().casefold() != other.strip().casefold()

        def cross_duplicate(overlap, box, other_box):
            # Containment alone may mean a person in a pool, not an alias.
            # Without masks, retain different categories rather than guessing.
            return overlap is not None and overlap[0] >= .85 and _iou(box, other_box) >= .70
        masks = [compact_mask(d.mask) if getattr(d, 'mask', None) is not None else None
                 for d in detections]
        candidates = []  # (same category, affinity, detection index, track ID)
        for di, det in enumerate(detections):
            for tid, tbox in active_boxes.items():
                same, iou = self.is_same_object(det.box, tbox)
                overlap = mask_overlap(masks[di], active_masks.get(tid))
                if overlap is not None:
                    # Disjoint masks are evidence against a box-only match.
                    # The tracker descriptor is one frame old; tolerate modest
                    # displacement but never use centroid proximity alone.
                    same = overlap[0] >= 0.30 or overlap[1] >= 0.90 or (
                        overlap[2] >= 0.90 and det.score >= 0.70 and
                        _centroid_dist_norm(det.box, tbox) <= 0.50) or (
                        iou >= 0.60 and _centroid_dist_norm(det.box, tbox) < 0.15)
                cross = different(det.prompt, active_prompts.get(tid))
                if cross:
                    same = cross_duplicate(overlap, det.box, tbox)
                if same:
                    candidates.append((not cross, max(iou, overlap[0] if overlap else 0), di, tid))
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

        matched_det = set()
        matched_track = set()
        matched: Dict[int, Detection] = {}
        matched_pairs: List[Tuple[int, int, float]] = []
        for _, iou, di, tid in candidates:
            if di in matched_det or tid in matched_track:
                continue
            matched_det.add(di)
            matched_track.add(tid)
            matched[tid] = detections[di]
            matched_pairs.append((tid, di, iou))

        # A one-to-one assignment is not duplicate rejection: another part of
        # an already matched vehicle must not automatically receive a new ID.
        suppressed = set()
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            for tid, tbox in active_boxes.items():
                overlap = mask_overlap(masks[di], active_masks.get(tid))
                cross = different(det.prompt, active_prompts.get(tid))
                duplicate = (cross_duplicate(overlap, det.box, tbox) if cross else
                             overlap is not None and overlap[1] >= 0.90 and _iou(det.box, tbox) > 0.05)
                if duplicate:
                    suppressed.add(di)
                    break
        # Prefer a supported whole-object mask over a contained part. Disjoint
        # adjacent masks are never joined solely because their boxes overlap.
        accepted = []
        # For aliases found in the same pass, prefer confidence, then prompt
        # order on ties. Same-category whole/part ordering remains unchanged.
        alias_winners = []
        for di in sorted(range(len(detections)), key=lambda i: (-detections[i].score, i)):
            if di in matched_det or di in suppressed:
                continue
            if any(different(detections[di].prompt, detections[j].prompt) and
                   cross_duplicate(mask_overlap(masks[di], masks[j]),
                                   detections[di].box, detections[j].box) for j in alias_winners):
                suppressed.add(di)
            else:
                alias_winners.append(di)
        order = sorted(range(len(detections)), key=lambda i: (
            int(masks[i].sum()) if masks[i] is not None else 0, detections[i].score), reverse=True)
        for di in order:
            if di in matched_det or di in suppressed:
                continue
            duplicate = False
            for other in list(matched_det) + accepted:
                overlap = mask_overlap(masks[di], masks[other])
                if different(detections[di].prompt, detections[other].prompt):
                    duplicate = cross_duplicate(overlap, detections[di].box, detections[other].box)
                elif overlap is not None:
                    duplicate = overlap[0] >= 0.70 or overlap[1] >= 0.90
                else:
                    duplicate = _iou(detections[di].box, detections[other].box) >= 0.85
                if duplicate:
                    break
            if duplicate:
                suppressed.add(di)
            else:
                accepted.append(di)
        new_detections = [detections[i] for i in accepted]
        return MatchResult(new_detections=new_detections, matched=matched,
                           matched_pairs=matched_pairs, suppressed=len(suppressed))
