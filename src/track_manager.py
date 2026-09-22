"""Track IDs, lifecycle validation, and bounded per-frame output recording.

New detections receive fresh external IDs; re-detection preserves existing SAM
memory. Lost rows carry null coordinates. Optional logical groups project member
tracks into shared output IDs without modifying their underlying SAM sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from detection_manager import DetectionManager
from sam31_video_runtime import Detection, SAM31VideoRuntime
from track_state import TrackState, TrackStateConfig, TrackStateMachine

COLUMNS = ["frame_number", "timestamp", "source_timestamp", "source_video",
           "track_id", "class_prompt", "confidence",
           "xmin", "ymin", "xmax", "ymax", "status"]

# `status` column values. Coordinates are written ONLY for VISIBLE rows; a lost
# track's row carries nulls rather than its stale last position, and a
# TERMINATED track gets exactly one transition row, then no more rows at all.
STATUS_VISIBLE = TrackState.VISIBLE.value
STATUS_TEMPORARILY_LOST = TrackState.TEMPORARILY_LOST.value
STATUS_OUT_OF_FRAME = TrackState.OUT_OF_FRAME.value
STATUS_TERMINATED = TrackState.TERMINATED.value


@dataclass
class FrameStats:
    frame_number: int
    kind: str            # "detect+track" | "track"
    n_active: int
    n_new: int
    track_latency_ms: float
    detect_latency_ms: float = 0.0


class TrackManager:
    def __init__(self, runtime: SAM31VideoRuntime, detection_manager: Optional[DetectionManager] = None,
                 max_objects: int = 16, logger=None, source_video: str = "",
                 state_config: Optional[TrackStateConfig] = None, bounded_history: bool = False,
                 group_config=None):
        self.runtime = runtime
        self.matcher = detection_manager or DetectionManager()
        self.log = logger or runtime.log
        # Lifecycle (VISIBLE / TEMPORARILY_LOST / OUT_OF_FRAME / TERMINATED) is
        # derived from the session's signals here; the session itself is not
        # changed and keeps its temporal memory for lost objects.
        self.state_config = state_config or TrackStateConfig()
        self.states = TrackStateMachine(self.state_config, logger=self.log)

        # The pool distributes tracks across fixed-capacity SAM sessions;
        # the overall object budget does not resize a session's multiplex.
        if isinstance(max_objects, bool) or int(max_objects) != max_objects or max_objects < 1:
            raise ValueError('max_objects must be a positive whole number')
        self.max_objects = max_objects
        if max_objects > 16:
            self.log.warning(
                "Maximum simultaneous objects: %d. Higher limits require more GPU memory "
                "and processing time. Reduce this value if GPU memory runs out or processing is too slow.",
                max_objects)
        # Recorded on every row so detections can be tied back to the original
        # file (and, for FMV, to its KLV/MISB metadata) after export.
        self.source_video = source_video

        self._next_id = 1
        self.bounded_history = bounded_history
        if bounded_history:
            from result_store import DiskHistory
            self.rows = DiskHistory(COLUMNS)
            self.frame_stats = DiskHistory()
        else:
            self.rows = []
            self.frame_stats = []
        self._terminated_count = 0
        self._retired_recoveries = 0
        self.track_prompt: Dict[int, str] = {}   # track_id -> the prompt that discovered it
        self._last_frame_row_count = 0
        self._trusted_masks = {}  # one compact mask per live ID
        self._duplicate_streaks = {}  # live ID pairs only
        from logical_groups import LogicalGroups
        self.logical_groups = LogicalGroups(group_config, self.log)
        self._group_detections = []
        self.columns = COLUMNS + (["member_track_ids"] if self.logical_groups.config.enabled else [])
        if bounded_history:
            self.rows.columns = self.columns

    # ------------------------------------------------------------------ ids
    @property
    def n_active(self) -> int:
        """Tracks whose id is still alive (VISIBLE or lost-within-grace)."""
        if not self.runtime.session.is_active:
            return 0
        return len([tid for tid in self.runtime.session.init_order
                    if (rec := self.states.records.get(tid)) is None or rec.alive])

    def _allocate_ids(self, n: int) -> List[int]:
        ids = list(range(self._next_id, self._next_id + n))
        self._next_id += n
        return ids

    def _active_boxes(self) -> Dict[int, List[float]]:
        """Last VALID box per live track, for duplicate suppression at re-detection.

        Lost tracks are included on purpose (their last visible position), so a
        detector hit on an object that is merely occluded does not spawn a
        second id. Terminated tracks are excluded: if that object turns up
        again it is, by definition, treated as new -- its old id is never
        reused for it or anything else.
        """
        sess = self.runtime.session
        if not sess.is_active:
            return {}
        boxes: Dict[int, List[float]] = {}
        for tid, box in sess.held.items():
            rec = self.states.records.get(tid)
            if rec is not None and not rec.alive:
                continue
            if rec is not None and rec.last_valid_box is not None:
                boxes[tid] = list(rec.last_valid_box)
            else:
                boxes[tid] = list(box[1:])
        return boxes

    def _slots_in_use(self) -> int:
        """Ids still registered in the SAM session (each costs a multiplex slot)."""
        sess = self.runtime.session
        return len(sess.init_order) if sess.is_active else 0

    # ------------------------------------------------------------------ steps
    def bootstrap_or_redetect(self, frame: np.ndarray, frame_idx: int, timestamp: float,
                              detections: List[Detection], detect_latency_ms: float,
                              source_timestamp: Optional[float] = None):
        """Handle a detection pass: match against active tracks, initialise only
        the genuinely new objects, then track this same frame."""
        t0 = time.time()
        if self.logical_groups.config.enabled:
            from track_quality import compact_mask
            self._group_detections = [(d.prompt, d.score, compact_mask(d.mask))
                                     for d in detections if getattr(d, 'mask', None) is not None]
        active_boxes = self._active_boxes()
        recent_masks = {tid: mask for tid, mask in self._trusted_masks.items()
                        if tid in active_boxes}
        result = self.matcher.match(detections, active_boxes, active_masks=recent_masks,
                                    active_prompts=self.track_prompt)
        if result.suppressed:
            self.log.info("frame %d: suppressed %d redundant detection(s)", frame_idx, result.suppressed)

        accepted = result.new_detections
        # Capacity is what the SAM session still holds, not what is alive: a
        # terminated track that was not released still occupies its slot.
        capacity = self.max_objects - self._slots_in_use()
        dropped = []
        if len(accepted) > capacity:
            accepted.sort(key=lambda d: d.score, reverse=True)
            accepted, dropped = accepted[:max(0, capacity)], accepted[max(0, capacity):]

        for tid, di, iou in result.matched_pairs:
            det = result.matched[tid]
            rec = self.states.records.get(tid)
            if rec is not None and det.score >= 0.70 and getattr(det, 'mask', None) is not None:
                rec.detection_support_box = list(det.box)
                rec.detection_support_frame = frame_idx
            self.log.debug("frame %d: detection matched existing track %d (affinity=%.2f, score=%.2f)",
                           frame_idx, tid, iou, det.score)
        if dropped:
            self.log.warning("frame %d: MAX_OBJECTS=%d reached; dropped %d low-confidence new detection(s)",
                             frame_idx, self.max_objects, len(dropped))

        new_rows = []
        if accepted:
            new_ids = self._allocate_ids(len(accepted))
            boxes_with_ids = []
            for tid, det in zip(new_ids, accepted):
                self.track_prompt[tid] = det.prompt
                boxes_with_ids.append([tid] + list(det.box))
            self.log.info("frame %d: %d new object(s) discovered by prompts %r -> track ids %s",
                          frame_idx, len(accepted), sorted({d.prompt for d in accepted}), new_ids)
            if not self.runtime.session.is_active:
                self.runtime.start_tracking(frame.shape[0], frame.shape[1])
            if all(getattr(det, 'mask', None) is not None for det in accepted):
                seeds = {tid: det.mask for tid, det in zip(new_ids, accepted)}
                new_rows = self.runtime.init_tracks(frame, boxes_with_ids, initial_masks=seeds)
            else:
                new_rows = self.runtime.init_tracks(frame, boxes_with_ids)
            for tid, det in zip(new_ids, accepted):
                self.states.register(tid, det.box, det.score, frame_idx)
        elif self.runtime.session.is_active and self.runtime.session.init_order:
            # No new objects, but this frame must still be tracked: add_objects
            # only propagates existing tracks when it has something to add, so
            # without this step every detection frame would report the PREVIOUS
            # frame's boxes and the lifecycle would count a phantom observation.
            self.runtime.track_frame(frame)

        n_new = len(accepted)
        self._record_frame(frame_idx, timestamp, kind="detect+track", latency_ms=(time.time() - t0) * 1000.0,
                           detect_latency_ms=detect_latency_ms, n_new=n_new,
                           source_timestamp=source_timestamp)
        return n_new

    def track_only(self, frame: np.ndarray, frame_idx: int, timestamp: float,
                   source_timestamp: Optional[float] = None):
        if not self.runtime.session.is_active or self.n_active == 0:
            self._record_frame(frame_idx, timestamp, kind="track", latency_ms=0.0, n_new=0,
                               source_timestamp=source_timestamp)
            return
        t0 = time.time()
        self.runtime.track_frame(frame)
        latency_ms = (time.time() - t0) * 1000.0
        self._record_frame(frame_idx, timestamp, kind="track", latency_ms=latency_ms, n_new=0,
                           source_timestamp=source_timestamp)

    def _mask_areas(self) -> Dict[int, int]:
        """Pixel area of each object's mask from the last tracker step, if available."""
        sess = self.runtime.session
        masks = getattr(sess, "last_masks", None)
        ids = list(getattr(sess, "last_obj_ids", []) or [])
        if masks is None or not ids:
            return {}

        if int(masks.shape[0]) != len(ids):
            raise RuntimeError("Tracker mask IDs and mask count disagree")
        # Match the area gate to the selected output component's box, rather
        # than allowing unrelated mask fragments elsewhere to satisfy it.
        import torch
        if tuple(masks.shape[-2:]) != (sess.frame_h, sess.frame_w):
            raise RuntimeError("Tracker masks must use video pixel dimensions")
        counts = []
        for i, oid in enumerate(ids):
            box = sess.held[int(oid)][1:]
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(sess.frame_w, int(np.ceil(box[2]))), min(sess.frame_h, int(np.ceil(box[3])))
            counts.append(masks[i, y1:y2, x1:x2].sum())
        areas = torch.stack(counts).tolist()
        return {int(oid): int(a) for oid, a in zip(ids, areas)}

    def _compact_current_masks(self):
        """Pool on the current device; transfer only bounded descriptors to CPU."""
        import torch
        sess = self.runtime.session
        masks = getattr(sess, 'last_masks', None)
        ids = list(getattr(sess, 'last_obj_ids', []) or [])
        if masks is None or not ids:
            return {}
        if masks.shape[0] != len(ids):
            raise RuntimeError("Tracker mask IDs and mask count disagree")
        h, w = masks.shape[-2:]
        scale = min(1.0, 384 / max(h, w))
        # Area resampling matches the detector descriptor's support convention.
        small = torch.nn.functional.interpolate(masks[:, None].float(),
            size=(max(1, round(h * scale)), max(1, round(w * scale))), mode='area')[:, 0]
        small = (small > 0).cpu().numpy()
        return dict(zip(map(int, ids), small))

    def _duplicate_tracks(self, masks, frame_idx):
        """Retire only persistent near-identical masks, never adjacent parts.

        A transient crossing cannot accumulate evidence from earlier encounters.
        Prefer the older confirmed ID. Geometrically implausible observations
        are handled by the lifecycle and cannot establish duplicate evidence.
        """
        from itertools import combinations
        from track_quality import mask_overlap, plausible_geometry
        from detection_manager import _iou
        sess = self.runtime.session
        streaks, retire = {}, set()
        eligible = []
        for tid in sess.init_order:
            rec = self.states.records.get(tid)
            if (rec is None or rec.state != TrackState.VISIBLE or
                    rec.last_accepted_frame != frame_idx or tid not in masks):
                continue
            box = sess.held[tid][1:]
            if plausible_geometry(rec.last_valid_box, box, frame_idx - rec.last_accepted_frame,
                                  rec.geometry_reference()):
                eligible.append(tid)
        for a, b in combinations(sorted(eligible), 2):
            overlap = mask_overlap(masks[a], masks[b])
            if overlap[0] < 0.85 or _iou(sess.held[a][1:], sess.held[b][1:]) < 0.70:
                continue
            pair = (a, b)
            streaks[pair] = self._duplicate_streaks.get(pair, 0) + 1
            if streaks[pair] >= 15:
                keep = min((a, b), key=lambda tid: (not self.states.records[tid].confirmed,
                                                   self.states.records[tid].born_at, tid))
                retire.add(b if keep == a else a)
        self._duplicate_streaks = streaks
        return retire

    def _record_frame(self, frame_idx, timestamp, kind, latency_ms, n_new, detect_latency_ms=0.0,
                      source_timestamp=None):
        sess = self.runtime.session
        before = len(self.rows)
        frame_rows = []
        if sess.is_active and sess.init_order:
            areas = self._mask_areas()
            masks = self._compact_current_masks()
            frame_w, frame_h = int(sess.frame_w), int(sess.frame_h)
            to_release: List[int] = []
            for tid in list(sess.init_order):
                held = sess.held.get(tid)
                if held is None:
                    continue
                score = float(sess.last_scores.get(tid, 0.0))
                rec = self.states.observe(
                    frame_idx, tid,
                    session_visible=bool(sess.visible.get(tid, False)),
                    score=score, held_box=list(held[1:]), mask_area=areas.get(int(tid)),
                    frame_w=frame_w, frame_h=frame_h)
            # Pair evidence must use final quality-accepted observations, not
            # raw session visibility before confidence/area/recovery gates.
            duplicates = self._duplicate_tracks(masks, frame_idx)
            for tid in list(sess.init_order):
                rec = self.states.records.get(tid)
                if rec is None:
                    continue
                score = float(sess.last_scores.get(tid, 0.0))
                if tid in duplicates and rec.alive:
                    rec.terminated_at = frame_idx
                    self.states._transition(rec, frame_idx, TrackState.TERMINATED,
                                            "persistent duplicate mask of established track")
                if rec.last_accepted_frame == frame_idx and rec.alive and tid in masks:
                    self._trusted_masks[tid] = masks[tid].copy()

                base = {
                    "frame_number": frame_idx, "timestamp": round(timestamp, 3),
                    "source_timestamp": (round(source_timestamp, 4)
                                          if source_timestamp is not None else None),
                    "source_video": self.source_video,
                    "track_id": int(tid), "class_prompt": self.track_prompt.get(tid, ""),
                }
                if rec.state == TrackState.VISIBLE:
                    x1, y1, x2, y2 = rec.last_valid_box
                    frame_rows.append({**base, "confidence": round(score, 4),
                                      "xmin": round(x1, 1), "ymin": round(y1, 1),
                                      "xmax": round(x2, 1), "ymax": round(y2, 1),
                                      "status": STATUS_VISIBLE})
                elif rec.state == TrackState.TERMINATED:
                    if rec.terminated_at == frame_idx:
                        # The transition, once. After this the track produces no rows.
                        frame_rows.append({**base, "confidence": None, "xmin": None, "ymin": None,
                                          "xmax": None, "ymax": None, "status": STATUS_TERMINATED})
                        if self.state_config.release_terminated_from_tracker:
                            to_release.append(int(tid))
                else:
                    # Lost: the id lives on, but its last position is NOT repeated
                    # as if it were a current observation.
                    frame_rows.append({**base, "confidence": round(score, 4), "xmin": None, "ymin": None,
                                      "xmax": None, "ymax": None, "status": rec.state.value})

            if self.logical_groups.config.enabled:
                for row in frame_rows:
                    row['member_track_ids'] = str(row['track_id'])
                frame_rows = self.logical_groups.project(frame_rows, masks, self.states.records,
                                                         timestamp, self._group_detections)
            if to_release:
                sess.remove_objects(to_release)
                for tid in to_release:
                    self._trusted_masks.pop(tid, None)
                self._duplicate_streaks = {pair: n for pair, n in self._duplicate_streaks.items()
                                          if not set(pair).intersection(to_release)}
                self.log.info("frame %d: released terminated track(s) %s from the SAM session",
                              frame_idx, to_release)
                if self.bounded_history:
                    for tid in to_release:
                        rec = self.states.records.pop(tid)
                        self._terminated_count += 1
                        self._retired_recoveries += rec.recoveries
                        self.track_prompt.pop(tid, None)

        self._group_detections = []
        for row in frame_rows:
            self.rows.append(row)
        if self.bounded_history:
            self.states.transitions[:] = [t for t in self.states.transitions if t.frame == frame_idx]

        self._last_frame_row_count = len(self.rows) - before
        self.frame_stats.append(FrameStats(frame_idx, kind, len(self.states.alive_ids()), n_new,
                                           latency_ms, detect_latency_ms))

    def last_frame_rows(self) -> pd.DataFrame:
        """The rows appended by the most recent bootstrap_or_redetect/track_only call.

        O(active tracks), not O(all frames so far) -- use this for per-frame
        drawing/preview instead of slicing results_dataframe() in a loop.
        """
        if self._last_frame_row_count == 0:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.rows[-self._last_frame_row_count:], columns=self.columns)

    # ------------------------------------------------------------------ output
    def results_dataframe(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(list(self.rows), columns=self.columns)

    def stats_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(s) for s in self.frame_stats])

    def performance_summary(self) -> dict:
        stats = self.frame_stats
        if not stats:
            return {}
        track_sum = track_count = detect_sum = detect_count = total_ms = passes = max_active = 0
        for s in stats:
            total_ms += s.track_latency_ms + s.detect_latency_ms
            passes += s.kind == "detect+track"
            max_active = max(max_active, s.n_active)
            if s.kind == "track" and s.track_latency_ms > 0:
                track_sum += s.track_latency_ms
                track_count += 1
            if s.detect_latency_ms > 0:
                detect_sum += s.detect_latency_ms
                detect_count += 1
        total_s = total_ms / 1000.0
        import torch
        gpu_mb = round(torch.cuda.memory_allocated() / 2**20) if torch.cuda.is_available() else None
        gpu_peak_mb = round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None
        return {
            "frames_processed": len(stats),
            "detection_passes": passes,
            "avg_track_latency_ms": round(track_sum / track_count, 1) if track_count else None,
            "avg_detect_latency_ms": round(detect_sum / detect_count, 1) if detect_count else None,
            "approx_fps": round(len(stats) / total_s, 2) if total_s > 0 else None,
            "max_active_tracks": max_active,
            "total_tracks_created": self._next_id - 1,
            "tracks_terminated": self._terminated_count + sum(1 for r in self.states.records.values() if not r.alive),
            "same_id_recoveries": self._retired_recoveries + sum(r.recoveries for r in self.states.records.values()),
            "gpu_allocated_mb": gpu_mb,
            "gpu_peak_mb": gpu_peak_mb,
        }

    def close_history(self):
        if self.bounded_history:
            self.rows.close()
            self.frame_stats.close()
