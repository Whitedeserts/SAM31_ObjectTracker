"""Track observation quality and lifecycle transitions.

Combines session visibility, confidence, valid mask area and geometry. Confirmed
tracks retain their IDs through bounded lost/out-of-frame grace periods. Only
accepted visible observations produce coordinates; termination is recorded once."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    VISIBLE = "VISIBLE"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    TERMINATED = "TERMINATED"


# States in which the object is not currently seen but its id is still alive.
LOST_STATES = (TrackState.TEMPORARILY_LOST, TrackState.OUT_OF_FRAME)


@dataclass
class TrackStateConfig:
    """Thresholds for lost/terminated handling. Defaults suit ~30 fps video."""

    lost_grace_frames: int = 30            # occluded: ~1 s of patience before terminating
    out_of_frame_grace_frames: int = 10    # left the frame: shorter, ~1/3 s
    min_valid_mask_area: int = 64          # pixels; below this a mask is not a visible object
    min_track_confidence: Optional[float] = None  # None = trust the tracker's own threshold
    edge_margin_px: int = 8                # box within this many px of an edge "touches" it
    hide_box_when_lost: bool = True        # never draw a box for a non-VISIBLE track
    show_lost_status: bool = True          # list lost tracks in the status panel instead
    release_terminated_from_tracker: bool = True  # remove terminated ids from the SAM session
    confirmation_frames: int = 3
    edge_confirmation_frames: int = 5
    recovery_frames: int = 2
    tentative_timeout: int = 10
    geometry_checks: bool = True
    recovery_confidence: float = 0.50

    def grace_for(self, state: TrackState) -> int:
        if state == TrackState.OUT_OF_FRAME:
            return int(self.out_of_frame_grace_frames)
        return int(self.lost_grace_frames)


@dataclass
class TrackRecord:
    track_id: int
    state: TrackState = TrackState.VISIBLE
    last_valid_box: Optional[List[float]] = None   # [x1, y1, x2, y2], only ever a VISIBLE box
    prev_valid_box: Optional[List[float]] = None   # the VISIBLE box before that, for motion
    last_score: float = 0.0
    missing_frames: int = 0
    lost_since: Optional[int] = None
    last_visible_frame: Optional[int] = None
    terminated_at: Optional[int] = None
    recoveries: int = 0
    exit_edges: Tuple[str, ...] = field(default_factory=tuple)
    born_at: int = 0
    confirmed: bool = False
    valid_streak: int = 0
    last_accepted_frame: Optional[int] = None
    last_reason: str = ""
    geometry_history: list = field(default_factory=list)  # at most 30 accepted boxes
    detection_support_box: Optional[List[float]] = None
    detection_support_frame: int = -10

    def geometry_reference(self):
        from track_quality import area
        return max(self.geometry_history or [self.last_valid_box], key=area)

    @property
    def alive(self) -> bool:
        return self.state != TrackState.TERMINATED


@dataclass
class Transition:
    frame: int
    track_id: int
    old: TrackState
    new: TrackState
    reason: str


# ---------------------------------------------------------------------------
# Box helpers (pure, reused by the drawing code)
def box_is_valid(box: Optional[Sequence[float]], frame_w: int, frame_h: int) -> bool:
    """Finite, positive area, and at least partly inside the frame."""
    if box is None or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return False
    if x2 <= 0 or y2 <= 0 or x1 >= frame_w or y1 >= frame_h:
        return False
    return True


def touched_edges(box: Sequence[float], frame_w: int, frame_h: int, margin: int) -> Tuple[str, ...]:
    x1, y1, x2, y2 = (float(v) for v in box)
    edges = []
    if x1 <= margin:
        edges.append("left")
    if y1 <= margin:
        edges.append("top")
    if x2 >= frame_w - margin:
        edges.append("right")
    if y2 >= frame_h - margin:
        edges.append("bottom")
    return tuple(edges)


def _moving_toward(prev: Sequence[float], last: Sequence[float], edges: Sequence[str]) -> Optional[bool]:
    """True/False if motion is known, None if there is no motion information."""
    pcx, pcy = (prev[0] + prev[2]) / 2.0, (prev[1] + prev[3]) / 2.0
    lcx, lcy = (last[0] + last[2]) / 2.0, (last[1] + last[3]) / 2.0
    dx, dy = lcx - pcx, lcy - pcy
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return None
    toward = {"left": dx < 0, "right": dx > 0, "top": dy < 0, "bottom": dy > 0}
    return any(toward[e] for e in edges)


# ---------------------------------------------------------------------------
class TrackStateMachine:
    """Owns one TrackRecord per id and advances them from the session's signals."""

    def __init__(self, config: Optional[TrackStateConfig] = None, logger=None):
        self.config = config or TrackStateConfig()
        self.log = logger
        self.records: Dict[int, TrackRecord] = {}
        self.transitions: List[Transition] = []

    # ------------------------------------------------------------------ input
    def register(self, track_id: int, box: Sequence[float], score: float, frame: int):
        """A detector seed needs subsequent observations before confirmation."""
        rec = TrackRecord(track_id=int(track_id), last_valid_box=[float(v) for v in box],
                          last_score=float(score), last_visible_frame=frame, born_at=frame,
                          last_accepted_frame=frame,
                          geometry_history=[[float(v) for v in box]],
                          confirmed=self.config.confirmation_frames <= 1,
                          state=(TrackState.VISIBLE if self.config.confirmation_frames <= 1
                                 else TrackState.TENTATIVE))
        self.records[rec.track_id] = rec
        return rec

    def observe(self, frame: int, track_id: int, *, session_visible: bool, score: float,
                held_box: Optional[Sequence[float]], mask_area: Optional[int],
                frame_w: int, frame_h: int) -> TrackRecord:
        """Advance one track by one frame from this frame's session signals."""
        rec = self.records.get(int(track_id))
        if rec is None:
            rec = self.register(track_id, held_box or [0, 0, 0, 0], score, frame)
        if not rec.alive:
            return rec

        rec.last_score = float(score)
        valid = self._valid_this_frame(session_visible, score, held_box, mask_area, frame_w, frame_h,
                                       continuing=rec.confirmed and rec.state == TrackState.VISIBLE)
        rec.last_reason = "invalid mask/score/box" if not valid else ""
        if valid and self.config.geometry_checks:
            from track_quality import plausible_geometry
            elapsed = frame - (rec.last_accepted_frame if rec.last_accepted_frame is not None else frame - 1)
            valid = plausible_geometry(rec.last_valid_box, held_box, elapsed, rec.geometry_reference())
            if not valid and frame - rec.detection_support_frame <= 2 and rec.detection_support_box:
                # A fresh, independently grounded detection can corroborate a
                # zoom/recovery that temporal geometry alone cannot explain.
                from track_quality import area
                support = rec.detection_support_box
                ratio = area(held_box) / max(1, area(support))
                valid = 0.25 <= ratio <= 2 and plausible_geometry(support, held_box)
            if not valid:
                rec.last_reason = "implausible geometry change"
        if valid and (not rec.confirmed or rec.state in LOST_STATES):
            valid = score >= max(self.config.recovery_confidence, self.config.min_track_confidence or 0)
            if not valid:
                rec.last_reason = "confirmation confidence below threshold"
        if valid:
            # Seed masks are prompts, not independent tracking evidence.
            if frame != rec.born_at:
                limit = max(1, self.config.confirmation_frames,
                            self.config.edge_confirmation_frames, self.config.recovery_frames)
                rec.valid_streak = min(limit, rec.valid_streak + 1)
            if not rec.confirmed:
                needed = (self.config.edge_confirmation_frames if touched_edges(
                    held_box, frame_w, frame_h, self.config.edge_margin_px)
                    else self.config.confirmation_frames)
                if rec.valid_streak >= needed:
                    rec.confirmed = True
                    self._transition(rec, frame, TrackState.VISIBLE, "track confirmed")
                rec.last_valid_box = list(held_box)
                rec.last_accepted_frame = frame
                rec.geometry_history = (rec.geometry_history + [list(held_box)])[-30:]
            if rec.confirmed and (rec.state not in LOST_STATES or
                                  rec.valid_streak >= self.config.recovery_frames):
                self._seen(rec, frame, held_box)
            elif rec.confirmed and rec.state in LOST_STATES:
                # The grace period counts elapsed frames, including valid
                # observations still awaiting recovery confirmation.
                rec.last_reason = "awaiting recovery confirmation"
                self._missed(rec, frame, frame_w, frame_h, session_visible, mask_area)
        else:
            rec.valid_streak = 0
            self._missed(rec, frame, frame_w, frame_h, session_visible, mask_area)
        if not rec.confirmed and rec.alive and frame - rec.born_at >= self.config.tentative_timeout:
            rec.terminated_at = frame
            self._transition(rec, frame, TrackState.TERMINATED, "tentative confirmation expired")
        return rec

    # ------------------------------------------------------------------ rules
    def _valid_this_frame(self, session_visible, score, box, mask_area, frame_w, frame_h,
                          continuing=False) -> bool:
        cfg = self.config
        if not session_visible:
            return False
        if cfg.min_track_confidence is not None and score < cfg.min_track_confidence:
            return False
        minimum_area = cfg.min_valid_mask_area * (0.75 if continuing else 1.0)
        if mask_area is not None and mask_area < minimum_area:
            return False
        return box_is_valid(box, frame_w, frame_h)

    def _seen(self, rec: TrackRecord, frame: int, box: Sequence[float]):
        if rec.state in LOST_STATES:
            rec.recoveries += 1
            self._transition(rec, frame, TrackState.VISIBLE,
                             f"recovered after {rec.missing_frames} missing frame(s)")
        rec.prev_valid_box, rec.last_valid_box = rec.last_valid_box, [float(v) for v in box]
        rec.missing_frames = 0
        rec.lost_since = None
        rec.last_visible_frame = frame
        rec.last_accepted_frame = frame
        rec.geometry_history = (rec.geometry_history + [list(box)])[-30:]
        rec.state = TrackState.VISIBLE

    def _missed(self, rec: TrackRecord, frame: int, frame_w: int, frame_h: int,
                session_visible: bool, mask_area: Optional[int]):
        rec.missing_frames += 1
        if rec.state == TrackState.VISIBLE:
            rec.lost_since = frame
            edges = touched_edges(rec.last_valid_box, frame_w, frame_h, self.config.edge_margin_px) \
                if rec.last_valid_box else ()
            heading_out = None
            if edges and rec.prev_valid_box is not None:
                heading_out = _moving_toward(rec.prev_valid_box, rec.last_valid_box, edges)
            if edges and heading_out is not False:
                rec.exit_edges = edges
                self._transition(rec, frame, TrackState.OUT_OF_FRAME,
                                 f"last box touched {'/'.join(edges)} edge"
                                 + (" moving outward" if heading_out else ""))
            else:
                why = rec.last_reason or ("score below threshold" if not session_visible else \
                    (f"mask area {mask_area} < {self.config.min_valid_mask_area}"
                     if mask_area is not None else "no valid box"))
                self._transition(rec, frame, TrackState.TEMPORARILY_LOST, why)

        if rec.missing_frames > self.config.grace_for(rec.state):
            rec.terminated_at = frame
            self._transition(rec, frame, TrackState.TERMINATED,
                             f"missing {rec.missing_frames} > grace {self.config.grace_for(rec.state)}")

    def _transition(self, rec: TrackRecord, frame: int, new: TrackState, reason: str):
        old = rec.state
        rec.state = new
        self.transitions.append(Transition(frame, rec.track_id, old, new, reason))
        if self.log:
            self.log.info("frame %d: track %d %s -> %s (%s)", frame, rec.track_id, old.value, new.value, reason)

    # ----------------------------------------------------------------- output
    def newly_terminated(self, frame: int) -> List[int]:
        return [t.track_id for t in self.transitions
                if t.frame == frame and t.new == TrackState.TERMINATED]

    def alive_ids(self) -> List[int]:
        return [tid for tid, rec in self.records.items() if rec.alive]

    def lost_records(self) -> List[TrackRecord]:
        return [rec for rec in self.records.values() if rec.state in LOST_STATES]
