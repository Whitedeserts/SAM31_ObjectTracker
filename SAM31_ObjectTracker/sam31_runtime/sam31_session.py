"""Persistent SAM 3.1 streaming tracking session for ArcGIS frame-by-frame calls.

Streaming design:

* One `inference_state` dict per tracking session, created by
  `Sam3VideoTrackingMultiplexDemo.init_state(video_height, video_width, num_frames)`.
  It holds the Object-Multiplex state, per-frame memories and object ids.
* Each ArcGIS frame t: run the shared ViT trunk once (`tracker.forward_image`),
  place the features in `state["cached_features"][t]`, grow `state["num_frames"]`
  to t+1, then advance the tracker exactly one frame with
  `tracker.propagate_in_video(state, start_frame_idx=t, max_frame_num_to_track=0)`.
  This is the same one-frame call the upstream multiplex orchestrator makes.
* New objects (ArcGIS boxes) are converted to masks and added with ONE batched
  `tracker.add_new_masks` call so all objects share the multiplex buckets.
* Memories older than `keep_window` frames are pruned so GPU memory is bounded;
  conditioning frames are never pruned.

Object removal: every initialization batch retains its encoded prompt frame as
conditioning memory, including batches added after tracking starts. This keeps
upstream removal from resetting all survivor memory when the original batch
leaves. Removal compacts the real multiplex state; no survivor is re-prompted
from a stale held box. Prompt frames are bounded by live input owners, and
are downgraded by upstream removal then reclaimed by normal window pruning.
Zero remaining objects still require a full lightweight state reset because
upstream removal returns before clearing its multiplex buckets. Models survive.
Output contract towards the ArcGIS host (EsriObjectTracker.dll):

* `step()` / `add_objects()` ALWAYS return exactly one row per initialised object
  id, in initialisation order, as plain Python floats:
  `[float(id), x1, y1, x2, y2]`, finite, inside the frame, width/height >= MIN_BOX.
* An object whose SAM 3.1 score is below `confidence_threshold` (occluded, out of
  view, lost) or whose mask is empty is reported with its LAST VALID box ("hold").
  It stays in SAM memory and is re-acquired with the same id when it reappears.
  Returning zero rows crashed ArcGIS Pro (the host indexes the result).
* No exception ever escapes to the host: on an internal failure the held boxes are
  returned, the traceback and a trace of the last frames are written to the log.

Nothing here reloads the model, re-creates the state, or replays frames.
"""

from __future__ import annotations

import gc
import math
import time
import traceback
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

from .arcgis_boxes import (
    MIN_BOX,
    ArcGISBox,
    box_to_sam_corner_points,
    masks_to_arcgis_boxes,
    sanitize_box,
)
from .frame_adapter import IMAGE_SIZE, frame_to_model_tensor
from .logging_utils import get_logger


class SessionError(RuntimeError):
    pass


class _CurrentFrameStore:
    """Stand-in for `inference_state["images"]`: only the current frame exists."""

    def __init__(self):
        self.frame_idx = None
        self.tensor = None

    def set(self, frame_idx, tensor):
        self.frame_idx, self.tensor = frame_idx, tensor

    def __getitem__(self, idx):
        if idx != self.frame_idx or self.tensor is None:
            raise SessionError(
                f"frame {idx} is not available (streaming session only holds frame {self.frame_idx})"
            )
        return self.tensor[0]

    def __len__(self):
        return 0 if self.frame_idx is None else self.frame_idx + 1


class SAM31StreamingSession:
    TRACE_LEN = 12

    def __init__(self, tracker, device="cuda", confidence_threshold=0.5, keep_window=32,
                 channel_order="RGB", logger=None):
        self.tracker = tracker
        self.device = torch.device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.keep_window = int(keep_window)
        self.channel_order = channel_order
        self.log = logger or get_logger()

        self.state: Optional[dict] = None
        self.frame_index = -1          # index of the last frame fed to the tracker
        self.frame_h = None
        self.frame_w = None
        # per-object bookkeeping (ArcGIS ids)
        self.init_order: List[int] = []            # ids in initialisation order (output order)
        self._mask_seeded_ids = set()
        self.held: Dict[int, List[float]] = {}     # last valid box per id [id, x1, y1, x2, y2]
        self.visible: Dict[int, bool] = {}
        self.lost_since: Dict[int, Optional[int]] = {}
        # last-frame diagnostics
        self.last_masks: Optional[torch.Tensor] = None   # (N, H, W) bool on GPU
        self.last_obj_ids: List[int] = []
        self.last_scores: Dict[int, float] = {}
        self.last_boxes: List[List[float]] = []          # full output (one row per id)
        self.last_detected: List[List[float]] = []       # rows for visible objects only
        self.trace = deque(maxlen=self.TRACE_LEN)
        self.stats = {"frames": 0, "track_steps": 0, "backbone_calls": 0,
                      "objects_added": 0, "errors": 0, "last_latency_ms": 0.0}
        self._autocast = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    # ------------------------------------------------------------------ lifecycle
    @property
    def is_active(self):
        return self.state is not None

    @property
    def object_ids(self):
        return list(self.init_order)

    def start(self, frame_h, frame_w):
        """Create a fresh inference state for a video of the given resolution."""
        self.close()
        with torch.inference_mode():
            self.state = self.tracker.init_state(
                video_height=int(frame_h), video_width=int(frame_w), num_frames=1
            )
        self.state["images"] = _CurrentFrameStore()
        self.frame_index = -1
        self.frame_h, self.frame_w = int(frame_h), int(frame_w)
        self.init_order, self.held, self.visible, self.lost_since = [], {}, {}, {}
        self.last_masks, self.last_obj_ids, self.last_scores = None, [], {}
        self.last_boxes, self.last_detected = [], []
        self.trace.clear()
        self.stats.update(frames=0, track_steps=0, backbone_calls=0, objects_added=0, errors=0)
        self.log.info("session started for %dx%d frames", frame_w, frame_h)

    def close(self, run_gc=True):
        """Free the session state (upstream close_session logic). Model survives."""
        if self.state is not None:
            self.state.clear()
            self.state = None
        self.last_masks = None
        self._mask_seeded_ids.clear()
        self.frame_index = -1
        if run_gc:
            gc.collect()
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                if total_b > 0 and (100.0 * (1 - free_b / total_b)) >= 80:
                    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ frames
    def _ingest_frame(self, frame, cached_features=None):
        """Preprocess + backbone for the next frame index. Returns t."""
        if self.state is None:
            raise SessionError("session not started: call init_tracker() first")
        if cached_features is None:
            img, h, w = frame_to_model_tensor(frame, self.device, IMAGE_SIZE, self.channel_order)
        else:
            img, backbone_out = cached_features
            h, w = frame.shape[:2]
        if (h, w) != (self.frame_h, self.frame_w):
            raise SessionError(
                f"frame size changed from {self.frame_w}x{self.frame_h} to {w}x{h}; "
                "start a new tracking session"
            )
        t = self.frame_index + 1
        from sam3.model.data_misc import NestedTensor

        if cached_features is None:
            with torch.inference_mode(), self._autocast():
                backbone_out = self.tracker.forward_image(
                    NestedTensor(tensors=img, mask=None),
                    need_sam3_out=False, need_interactive_out=True, need_propagation_out=True,
                )
            self.stats["backbone_calls"] += 1
        self.state["cached_features"] = {t: (img, backbone_out)}
        self.state["images"].set(t, img)
        self.state["num_frames"] = t + 1
        self.frame_index = t
        return t

    def _propagate_one(self, t):
        """Advance tracker to frame t (already ingested). Returns (obj_ids, masks_bool, scores)."""
        if self.init_order and not self._cond_frame_count():
            raise SessionError("live objects have no conditioning memory")
        outputs = None
        with torch.inference_mode(), self._autocast():
            for out in self.tracker.propagate_in_video(
                self.state, start_frame_idx=t, max_frame_num_to_track=0, reverse=False,
                tqdm_disable=True, run_mem_encoder=True,
            ):
                outputs = out
        if outputs is None:
            raise SessionError(f"tracker produced no output for frame {t}")
        frame_idx, obj_ids, _low_res, video_res_masks, obj_scores = outputs
        if frame_idx != t:
            raise SessionError(f"tracker returned frame {frame_idx}, expected {t}")
        self.stats["track_steps"] += 1
        if video_res_masks is None or video_res_masks.numel() == 0:
            masks = torch.zeros(0, self.frame_h, self.frame_w, dtype=torch.bool, device=self.device)
        else:
            masks = (video_res_masks > 0.0).squeeze(1)          # (N, H, W) bool
        scores = torch.sigmoid(obj_scores.float().flatten()) if obj_scores is not None \
            else torch.zeros(len(obj_ids), device=self.device)  # (N,)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        return [int(o) for o in obj_ids], masks, scores

    def _prune(self, t):
        cutoff = t - self.keep_window
        if cutoff <= 0:
            return
        od = self.state["output_dict"]["non_cond_frame_outputs"]
        for k in [k for k in od if k < cutoff]:
            del od[k]
        for per_obj in self.state["output_dict_per_obj"].values():
            d = per_obj["non_cond_frame_outputs"]
            for k in [k for k in d if k < cutoff]:
                del d[k]
        fat = self.state["frames_already_tracked"]
        for k in [k for k in fat if k < cutoff]:
            del fat[k]

    # ------------------------------------------------------------- conditioning / removal
    def _cond_frame_count(self) -> int:
        return len(self.state["output_dict"]["cond_frame_outputs"])

    def _retain_prompt_frame(self, t):
        """Retain the actual initialization memory of every added batch.

        Existing objects must first propagate to t for multiplex merging, so
        SAM classifies a later batch as non-conditioning. Promote its already
        encoded output, including per-object views and consolidated indices,
        rather than re-prompting survivors after removal destroys their memory.
        Each retained frame has a live input owner; upstream removal downgrades
        it when its last owner leaves, and normal window pruning reclaims it.
        """
        state = self.state
        for outputs in [state["output_dict"], *state["output_dict_per_obj"].values()]:
            out = outputs["non_cond_frame_outputs"].pop(t, None)
            if out is not None:
                outputs["cond_frame_outputs"][t] = out
        indices = state["consolidated_frame_inds"]
        indices["non_cond_frame_outputs"].discard(t)
        indices["cond_frame_outputs"].add(t)

    def _conditioning_diagnostic(self) -> dict:
        """Snapshot used for the removal log line and for tests."""
        mux = self.state.get("multiplex_state") if self.state else None
        return {
            "active_ids": list(self.init_order),
            "valid_entries": int(getattr(mux, "total_valid_entries", 0) or 0),
            "bucket_count": int(getattr(mux, "num_buckets", 0) or 0),
            "conditioning_frames": sorted(self.state["output_dict"]["cond_frame_outputs"].keys())
                                   if self.state else [],
        }

    # ------------------------------------------------------------------ output contract
    def held_output(self) -> List[List[float]]:
        """One sanitised row per initialised id from the held boxes (always host-safe)."""
        rows = []
        for oid in self.init_order:
            b = self.held.get(oid)
            if b is None:
                continue
            row = sanitize_box(b, self.frame_w, self.frame_h)
            if row is not None:
                rows.append(row)
        return rows

    def _finish(self, t, obj_ids, masks, scores, t0):
        """Turn SAM results into the ArcGIS output; never raises for lost objects."""
        n = len(obj_ids)
        if scores.numel() != n or masks.shape[0] != n:
            # inconsistent state: report but keep the host alive with held boxes
            self.log.error("frame %d: tracker returned %d masks / %d scores for %d objects; holding boxes",
                           t, int(masks.shape[0]), int(scores.numel()), n)
            self.stats["errors"] += 1
            scores = torch.zeros(n, device=masks.device)
            masks = torch.zeros(n, self.frame_h, self.frame_w, dtype=torch.bool, device=masks.device)

        scores_cpu = scores.detach().float().cpu().tolist()
        self.last_scores = {oid: float(s) for oid, s in zip(obj_ids, scores_cpu)}
        keep = [i for i, s in enumerate(scores_cpu) if s >= self.confidence_threshold]
        detected: Dict[int, List[float]] = {}
        if keep:
            idx = torch.as_tensor(keep, device=masks.device)
            for row in masks_to_arcgis_boxes([obj_ids[i] for i in keep], masks.index_select(0, idx),
                                              join_component_ids=self._mask_seeded_ids):
                row = sanitize_box(row, self.frame_w, self.frame_h)
                if row is not None:
                    detected[int(row[0])] = row

        out: List[List[float]] = []
        for oid in self.init_order:
            if oid in detected:
                row = detected[oid]
                self.held[oid] = list(row)
                if not self.visible.get(oid, True):
                    self.log.info("frame %d: object %d re-acquired after %d lost frame(s) (score %.3f)",
                                  t, oid, t - (self.lost_since.get(oid) or t), self.last_scores.get(oid, 0.0))
                self.visible[oid] = True
                self.lost_since[oid] = None
            else:
                row = self.held.get(oid)
                if row is None:  # should not happen: every id gets a held box at init
                    continue
                if self.visible.get(oid, True):
                    self.log.info("frame %d: object %d lost (score %.3f, mask empty=%s); holding last box %s",
                                  t, oid, self.last_scores.get(oid, 0.0),
                                  oid not in detected and self.last_scores.get(oid, 0.0) >= self.confidence_threshold,
                                  [round(v, 1) for v in row[1:]])
                    self.lost_since[oid] = t
                self.visible[oid] = False
                row = sanitize_box(row, self.frame_w, self.frame_h)
            if row is not None:
                out.append([float(v) for v in row])

        self.last_masks = masks
        self.last_obj_ids = list(obj_ids)
        self.last_detected = list(detected.values())
        self.last_boxes = out
        self._prune(t)
        self.stats["frames"] += 1
        self.stats["last_latency_ms"] = (time.time() - t0) * 1000.0
        self._record(t, obj_ids, scores_cpu, masks, out)
        if self.log.isEnabledFor(10):
            self.log.debug("frame %d: %d objects, %d visible, %d rows out, %.1f ms",
                           t, n, len(detected), len(out), self.stats["last_latency_ms"])
        return out

    def _record(self, t, obj_ids, scores, masks, out):
        try:
            areas = masks.flatten(1).sum(1).tolist() if masks.numel() else []
        except Exception:
            areas = []
        self.trace.append({
            "frame": t, "ids": list(obj_ids), "scores": [round(s, 4) for s in scores],
            "mask_areas": [int(a) for a in areas], "visible": [bool(self.visible.get(o, False)) for o in obj_ids],
            "out": [[round(v, 1) for v in r] for r in out], "ms": round(self.stats["last_latency_ms"], 1),
        })

    def dump_trace(self, reason):
        """Write the last frames' state to the log (called on failures)."""
        self.log.error("---- SAM31 trace dump (%s) ----", reason)
        self.log.error("frame_index=%s size=%sx%s ids=%s held=%s visible=%s multiplex=%s stats=%s",
                       self.frame_index, self.frame_w, self.frame_h, self.init_order,
                       {k: [round(x, 1) for x in v[1:]] for k, v in self.held.items()},
                       self.visible, self.multiplex_info(), self.stats)
        for rec in self.trace:
            self.log.error("  %s", rec)
        self.log.error("---- end trace ----")

    def _fail(self, where, exc, t0):
        """Log an internal failure with traceback + trace and return the held output."""
        self.stats["errors"] += 1
        self.visible = {oid: False for oid in self.init_order}
        self.last_scores = {oid: 0.0 for oid in self.init_order}
        self.last_masks, self.last_obj_ids, self.last_detected = None, [], []
        self.log.error("%s failed at frame %d: %s\n%s", where, self.frame_index, exc, traceback.format_exc())
        self.dump_trace(f"{where} exception")
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.stats["last_latency_ms"] = (time.time() - t0) * 1000.0
        out = self.held_output()
        self.last_boxes = out
        return out

    # ------------------------------------------------------------------ prompts
    def _boxes_to_masks(self, t, boxes: List[ArcGISBox]):
        """ArcGIS boxes -> (N, H, W) bool masks on frame t via the SAM 3.1 interactive head.

        Uses a throw-away inference state that shares this session's feature cache,
        so the backbone is not re-run. Each box is a SAM box-corner prompt
        (labels 2/3) on an initial conditioning frame (no memory involved).
        If SAM returns an empty mask for a box, the box itself is rasterised as the
        mask so that the object still enters the tracker with a valid memory.
        """
        scratch = self.tracker.init_state(
            video_height=self.frame_h, video_width=self.frame_w, num_frames=t + 1,
            cached_features=self.state["cached_features"],
        )
        scratch["images"] = self.state["images"]
        masks = []
        try:
            for box in boxes:
                pts, labels = box_to_sam_corner_points(box, self.frame_w, self.frame_h)
                _, _, _, video_res = self.tracker.add_new_points(
                    scratch, t, int(box[0]), pts, labels, clear_old_points=True, rel_coordinates=True
                )
                m = video_res[0, 0] > 0.0
                if not bool(m.any()):
                    self.log.warning("frame %d: SAM gave an empty mask for box of id %d; using the box as mask",
                                     t, int(box[0]))
                    m = torch.zeros(self.frame_h, self.frame_w, dtype=torch.bool, device=self.device)
                    _, x1, y1, x2, y2 = box
                    m[int(y1):max(int(y1) + 1, int(math.ceil(y2))), int(x1):max(int(x1) + 1, int(math.ceil(x2)))] = True
                masks.append(m)
        finally:
            scratch.clear()
        return torch.stack(masks, dim=0)  # (N, H, W) bool

    # ------------------------------------------------------------------ API
    @staticmethod
    def validate_initial_masks(boxes, masks, height, width):
        """Validate detector seeds before any ID/session mutation."""
        ids = {int(b[0]) for b in boxes}
        if set(masks) != ids:
            raise ValueError("Initial masks must have exactly the new object IDs")
        for mask in masks.values():
            if tuple(mask.shape) != (height, width):
                raise ValueError("Initial mask dimensions must match the video frame")
            data = torch.as_tensor(mask)
            if data.dtype != torch.bool or not bool(data.any()):
                raise ValueError("Initial masks must be nonempty boolean masks")

    def add_objects(self, frame, boxes: List[ArcGISBox], cached_features=None, initial_masks=None):
        """Feed `frame` as the next frame and initialise the given objects on it.

        Follows the upstream multiplex orchestrator (`_tracker_add_new_objects`):
        all new objects of a frame are added with ONE `add_new_masks` call so they
        are packed into shared multiplex buckets (existing free slots first, new
        buckets only when needed), then `propagate_in_video_preflight` encodes
        their memory. Existing objects are propagated to this frame first.
        Ids that already exist are re-initialised (removed and re-added with the
        new box), which is the ArcGIS "Move Object" semantics.
        Always returns one row per initialised id (see module docstring).
        """
        t0 = time.time()
        if self.state is None:
            raise SessionError("session not started")
        if not boxes:
            return self.held_output()
        if initial_masks is not None:
            self.validate_initial_masks(boxes, initial_masks, self.frame_h, self.frame_w)
            self._mask_seeded_ids.update(initial_masks)
        else:
            self._mask_seeded_ids.difference_update(int(b[0]) for b in boxes)
        # register ids + held boxes first so that even a failure below yields valid rows
        for b in boxes:
            oid = int(b[0])
            if oid not in self.init_order:
                self.init_order.append(oid)
            self.held[oid] = [float(oid), float(b[1]), float(b[2]), float(b[3]), float(b[4])]
            self.visible[oid] = True
            self.lost_since[oid] = None
        try:
            t = self._ingest_frame(frame, cached_features)
            existing = set(self.state["obj_ids"])
            reinit = [int(b[0]) for b in boxes if int(b[0]) in existing]
            if existing:
                # propagate the existing objects to this frame first so that the new
                # objects are merged into an output that exists for frame t
                self._propagate_one(t)

            with torch.inference_mode(), self._autocast():
                if reinit:
                    self.tracker.remove_objects(self.state, obj_ids=reinit, strict=False, need_output=False)
                    if not self.state["obj_ids"]:
                        # every tracked object is being re-initialised: start from a clean state
                        self.tracker.clear_all_points_in_video(self.state)
                    self.log.info("frame %d: re-initialising existing id(s) %s", t, reinit)
                if initial_masks is None:
                    masks = self._boxes_to_masks(t, boxes)
                else:
                    # Text grounding already segmented the target. Preserve its
                    # meaning instead of asking the box-prompt head to segment again.
                    masks = torch.stack([torch.as_tensor(initial_masks[int(b[0])],
                                          device=self.device) for b in boxes])
                obj_ids = [int(b[0]) for b in boxes]
                self.tracker.add_new_masks(
                    inference_state=self.state, frame_idx=t, obj_ids=obj_ids,
                    masks=masks.float(), add_mask_to_memory=True,
                )
                self.tracker.propagate_in_video_preflight(self.state, run_mem_encoder=True)
                self.stats["objects_added"] += len(boxes)
                self._retain_prompt_frame(t)

            all_ids, masks, scores = self._propagate_one(t)   # fetches consolidated outputs for t
            mux = self.state.get("multiplex_state")
            self.log.info(
                "frame %d: %d object(s) initialised (%s), %d tracked; multiplex buckets=%s valid_entries=%s capacity=%s",
                t, len(boxes), "new session" if not existing else "added to live session", len(all_ids),
                getattr(mux, "num_buckets", None), getattr(mux, "total_valid_entries", None),
                getattr(mux, "allowed_bucket_capacity", None),
            )
            return self._finish(t, all_ids, masks, scores, t0)
        except Exception as e:  # noqa: BLE001 - never let it reach the host
            return self._fail("init_tracker", e, t0)

    def step(self, frame, cached_features=None):
        """Track all objects on the next frame. Always returns one row per id."""
        t0 = time.time()
        if self.state is None or not self.init_order:
            return []
        try:
            t = self._ingest_frame(frame, cached_features)
            obj_ids, masks, scores = self._propagate_one(t)
            return self._finish(t, obj_ids, masks, scores, t0)
        except Exception as e:  # noqa: BLE001 - never let it reach the host
            return self._fail("track", e, t0)

    def remove_objects(self, obj_ids):
        """Permanently drop `obj_ids` (e.g. TERMINATED tracks) from SAM memory.

        Removes live multiplex entries while preserving survivors' actual
        conditioning and temporal memory, with their external ids unchanged.
        """
        if self.state is None:
            return
        obj_ids = list(dict.fromkeys(int(o) for o in obj_ids))
        before = self._conditioning_diagnostic()
        with torch.inference_mode(), self._autocast():
            self.tracker.remove_objects(self.state, obj_ids=obj_ids, strict=False, need_output=False)
        for oid in obj_ids:
            if oid in self.init_order:
                self.init_order.remove(oid)
            self.held.pop(oid, None)
            self.visible.pop(oid, None)
            self.lost_since.pop(oid, None)
            self.last_scores.pop(oid, None)
            self._mask_seeded_ids.discard(oid)
        # Keep mask indices aligned with their IDs immediately after removal.
        # Consumers may read diagnostics before the next propagation call.
        removed = set(obj_ids)
        keep = [i for i, oid in enumerate(self.last_obj_ids) if oid not in removed]
        if self.last_masks is not None:
            self.last_masks = self.last_masks[keep] if keep else None
        self.last_obj_ids = [self.last_obj_ids[i] for i in keep]
        self.last_boxes = [row for row in self.last_boxes if int(row[0]) not in removed]
        self.last_detected = [row for row in self.last_detected if int(row[0]) not in removed]
        after = self._conditioning_diagnostic()
        self.log.info(
            "remove_objects: before active_ids=%s valid_entries=%s | remove=%s | "
            "after active_ids=%s valid_entries=%s bucket_count=%s conditioning_frames=%s",
            before["active_ids"], before["valid_entries"], obj_ids,
            after["active_ids"], after["valid_entries"], after["bucket_count"],
            after["conditioning_frames"])

        if not self.init_order:
            # Upstream returns before clearing multiplex buckets when the last ID
            # is removed. Reset lightweight state so later additions cannot attach
            # to orphaned buckets; model weights remain loaded.
            with torch.inference_mode(), self._autocast():
                self.tracker.clear_all_points_in_video(self.state)
            self.log.info("frame %d: all tracks removed; SAM state fully reset "
                          "(waiting for the next detection pass)", self.frame_index)
            self.last_masks, self.last_obj_ids = None, []
            self.last_boxes, self.last_detected = [], []
        elif not after["conditioning_frames"]:
            raise SessionError("removal left live objects without conditioning memory")

    def multiplex_info(self):
        mux = self.state.get("multiplex_state") if self.state else None
        if mux is None:
            return {"num_buckets": 0, "total_valid_entries": 0, "capacity": None}
        return {"num_buckets": int(mux.num_buckets), "total_valid_entries": int(mux.total_valid_entries),
                "capacity": int(mux.allowed_bucket_capacity)}

