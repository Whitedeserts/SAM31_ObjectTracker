"""Optional articulated-vehicle output grouping; never modifies SAM sessions."""
from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from track_quality import mask_overlap, mask_box


@dataclass(frozen=True)
class GroupConfig:
    enabled: bool = False
    confirm_seconds: float = 0.5
    release_seconds: float = 1.0
    evidence_seconds: float = 1.5
    relative_motion: float = 0.9  # diagonals/second; 0.03/frame at 30 fps
    relative_spread: float = 0.10
    max_gap: float = 0.10

    def __post_init__(self):
        for name in ('confirm_seconds', 'release_seconds', 'evidence_seconds',
                     'relative_motion', 'relative_spread', 'max_gap'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')


def geometry(mask):
    """Principal axis and bounds from an already bounded mask descriptor."""
    y, x = np.nonzero(mask)
    if len(x) < 8:
        return None
    points = np.column_stack((x, y)).astype(float)
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points - center, rowvar=False))
    if values[0] <= 0 or values[1] / values[0] < 2:
        return None  # orientation of near-round/tiny masks is unreliable
    return center, vectors[:, 1], points, mask_box(mask)


def attachment(a, b, config, descriptors=None):
    """Require approximately collinear, end-to-end support, not lateral overlap."""
    ga, gb = descriptors if descriptors is not None else (geometry(a), geometry(b))
    if ga is None or gb is None:
        return None
    ca, axis, pa, ba = ga
    cb, other_axis, pb, bb = gb
    if abs(axis @ other_axis) < math.cos(math.radians(25)):
        return None
    normal = np.array([-axis[1], axis[0]])
    widths = [np.ptp(p @ normal) + 1 for p in (pa, pb)]
    if abs((cb - ca) @ normal) > .35 * min(widths):
        return None
    ranges = [np.percentile(p @ axis, [2, 98]) for p in (pa, pb)]
    lengths = [r[1] - r[0] for r in ranges]
    gap = max(ranges[0][0], ranges[1][0]) - min(ranges[0][1], ranges[1][1])
    diagonal = max(8., sum(math.hypot(box[2]-box[0], box[3]-box[1])
                           for box in (ba, bb)) / 2)
    if gap > config.max_gap * diagonal or gap < -.25 * min(lengths):
        return None
    return (cb - ca) / diagonal


def whole_support(whole, a, b):
    """Reject oversized scene masks and require both component masks covered."""
    union = a | b
    overlap = mask_overlap(whole, union)
    return (overlap is not None and overlap[0] >= .70
            and mask_overlap(whole, a)[2] >= .80
            and mask_overlap(whole, b)[2] >= .80)


def translated(mask, dx, dy):
    """Integer translation with zero padding, never wrap support at frame edges."""
    h, w = mask.shape
    result = np.zeros_like(mask)
    if abs(dx) >= w or abs(dy) >= h:
        return result
    result[max(0, dy):min(h, h+dy), max(0, dx):min(w, w+dx)] = \
        mask[max(0, -dy):min(h, h-dy), max(0, -dx):min(w, w-dx)]
    return result


class LogicalGroups:
    def __init__(self, config=None, logger=None):
        self.config = config or GroupConfig()
        self.log = logger
        self.candidates = {}
        self.groups = {}  # surviving logical ID -> original pair
        self.separation = {}
        self.whole = {}  # at most one recent reference mask per live member
        self.evidence = {}  # live pairs with recent whole-object corroboration
        self.last_time = None
        self.cadence = deque(maxlen=30)
        self.previous_offsets = {}

    def project(self, rows, masks, records, timestamp, detections=()):
        """Project validated member rows into logical rows, with bounded state."""
        if not self.config.enabled:
            return rows
        now = float(timestamp)
        dt = 0 if self.last_time is None else now - self.last_time
        self.last_time = now
        if not math.isfinite(now) or dt < 0:
            raise ValueError('Grouping requires finite, nondecreasing video timestamps')
        gap = bool(self.cadence and dt > 3 * np.median(self.cadence))
        if dt > 0 and not gap:
            self.cadence.append(dt)
        if gap:
            self.candidates.clear()
        observed_dt = 0 if gap else dt
        by_id = {r['track_id']: r for r in rows}
        alive = {tid for tid, r in records.items() if r.alive}
        visible = {tid: masks[tid] for tid, row in by_id.items()
                   if row['status'] == 'VISIBLE' and tid in masks}
        descriptors = {tid: geometry(mask) for tid, mask in visible.items()}
        self.whole = {tid: item for tid, item in self.whole.items()
                      if tid in alive and now-item[0] <= self.config.evidence_seconds}
        self.evidence = {p: t for p, t in self.evidence.items()
                         if set(p) <= alive and now-t <= self.config.evidence_seconds}
        current = {}
        for pair in combinations(sorted(visible), 2):
            a, b = pair
            if by_id[a]['class_prompt'].strip().casefold() != by_id[b]['class_prompt'].strip().casefold():
                continue
            offset = attachment(visible[a], visible[b], self.config,
                                (descriptors[a], descriptors[b]))
            if offset is None:
                continue
            current[pair] = offset
            for prompt, score, mask in detections:
                if (score >= .70 and prompt.strip().casefold() == by_id[a]['class_prompt'].strip().casefold()
                        and whole_support(mask, visible[a], visible[b])):
                    self.evidence[pair] = now
            union = visible[a] | visible[b]
            union_box = mask_box(union)
            for tid in pair:
                old = self.whole.get(tid)
                if old is None or visible[tid].sum() > .75 * old[1].sum():
                    continue
                old_box = mask_box(old[1])
                shift = np.rint((np.array(union_box[:2])+union_box[2:]
                                -np.array(old_box[:2])-old_box[2:])/2).astype(int)
                if whole_support(translated(old[1], *shift), visible[a], visible[b]):
                    self.evidence[pair] = now
        # Keep a larger recent mask long enough to recognize a split; expiry
        # prevents a historical zoom scale from becoming permanent evidence.
        for tid, mask in visible.items():
            if tid not in self.whole or mask.sum() >= self.whole[tid][1].sum():
                self.whole[tid] = (now, mask.copy())

        moving_apart = set()
        for pair, offset in current.items():
            previous = self.previous_offsets.get(pair)
            if previous is not None and observed_dt > 0:
                if np.linalg.norm(offset-previous) / observed_dt > self.config.relative_motion:
                    moving_apart.add(pair)
        self.previous_offsets = current
        used = set()
        ended = []
        for parent, pair in list(self.groups.items()):
            living = set(pair) & alive
            if not living:
                ended.append(parent)
                continue
            if set(pair) <= visible.keys():
                bad = pair not in current or pair in moving_apart
                self.separation[parent] = self.separation.get(parent, 0.) + observed_dt if bad else 0.
                if self.separation[parent] >= self.config.release_seconds:
                    self.groups.pop(parent)
                    self.separation.pop(parent, None)
                    if self.log:
                        self.log.info('Vehicle group %s separated; restored member IDs %s', parent, pair)
                    continue
            used.update(pair)

        candidates = {}
        for pair, offset in current.items():
            if set(pair) & used or pair not in self.evidence or pair in moving_apart or gap:
                continue
            history = self.candidates.get(pair, deque(maxlen=60))
            # Sample in time so a bounded deque spans the confirmation window
            # even at high FPS. Every observation still passes the motion gate.
            if not history or now-history[-1][0] >= max(1., self.config.confirm_seconds) / 58:
                history.append((now, offset))
            while len(history) > 2 and now-history[0][0] > max(1., self.config.confirm_seconds) + dt:
                history.popleft()
            if np.max(np.ptp(np.array([x[1] for x in history]), axis=0)) > self.config.relative_spread:
                history = deque([(now, offset)], maxlen=60)
            candidates[pair] = history
            if len(history) >= 3 and now-history[0][0] >= self.config.confirm_seconds:
                parent = min(pair, key=lambda tid: (records[tid].born_at, tid))
                self.groups[parent] = pair
                used.update(pair)
                if self.log:
                    self.log.info('Vehicle group %s confirmed from member IDs %s', parent, pair)
        self.candidates = {p: h for p, h in candidates.items() if not set(p) & used}

        projected = dict(by_id)
        for parent, pair in self.groups.items():
            members = [by_id[tid] for tid in pair if tid in by_id]
            if not members:
                continue
            shown = [r for r in members if r['status'] == 'VISIBLE']
            live_rows = [r for r in members if r['track_id'] in alive]
            row = dict((shown or live_rows or members)[0])
            row['track_id'] = parent
            row['member_track_ids'] = ','.join(map(str, pair))
            if shown:
                # Bounds of the union of current accepted component extents;
                # never union stale/lost masks or feed this back into SAM.
                for key, fn in (('xmin', min), ('ymin', min), ('xmax', max), ('ymax', max)):
                    row[key] = fn(r[key] for r in shown)
                row['confidence'] = min(r['confidence'] for r in shown)
            for tid in pair:
                projected.pop(tid, None)
            projected[parent] = row
        for parent in ended:
            self.groups.pop(parent)
            self.separation.pop(parent, None)
        return list(projected.values())
