"""Bounded SAM multiplex sessions sharing one model and one backbone per frame.

SAM tombstones removed slots because historical memory contains their slot
identity. Never recycle those physical slots or resize an old bucket's history.
New tracks use unused slots or a fresh session. Empty sessions are discarded;
there can never be more sessions than live objects. Existing tracks stay in
their original sessions with all temporal memory and external IDs intact.
"""
import torch
from sam31_runtime.sam31_session import SAM31StreamingSession, SessionError


class StreamingSessionPool:
    def __init__(self, tracker, **options):
        self.tracker, self.options = tracker, options
        self.sessions = []
        self.is_active = False
        self.init_order = []
        self.frame_h = self.frame_w = None
        self.frame_index = -1
        self.stats = {"errors": 0, "backbone_calls": 0}

    def start(self, frame_h, frame_w):
        self.close()
        self.frame_h, self.frame_w = frame_h, frame_w
        self.is_active = True
        self.frame_index = -1
        self.stats = {"errors": 0, "backbone_calls": 0}

    def close(self):
        for session in self.sessions:
            session.close(run_gc=False)
        self.sessions = []
        self.init_order = []
        self.is_active = False

    def _combined(self, name):
        return {key: value for session in self.sessions for key, value in getattr(session, name).items()}

    @property
    def held(self):
        return self._combined("held")

    @property
    def visible(self):
        return self._combined("visible")

    @property
    def last_scores(self):
        return self._combined("last_scores")

    @property
    def last_obj_ids(self):
        return [oid for session in self.sessions for oid in session.last_obj_ids]

    @property
    def last_masks(self):
        masks = [session.last_masks for session in self.sessions if session.last_masks is not None]
        return torch.cat(masks) if masks else None

    def held_output(self):
        boxes = self.held
        return [boxes[oid] for oid in self.init_order]

    def add_objects(self, frame, boxes, initial_masks=None):
        if not self.is_active:
            raise SessionError("session pool not started")
        # Validate the complete batch before registering IDs or opening sessions.
        # A rejected batch must not leave an ID without corresponding SAM state.
        boxes = list(boxes)
        if initial_masks is not None:
            SAM31StreamingSession.validate_initial_masks(
                boxes, initial_masks, self.frame_h, self.frame_w)
        seen = set(self.init_order)
        for box in boxes:
            oid = int(box[0])
            if oid in seen:
                raise SessionError(f"track {oid} already exists or is repeated in the batch")
            seen.add(oid)
        batches = {session: [] for session in self.sessions}
        for box in boxes:
            oid = int(box[0])
            if oid in self.init_order:
                # Toolbox detection only introduces new external IDs.
                raise SessionError(f"track {oid} already exists in the toolbox session pool")
            target = None
            for session in self.sessions:
                mux = session.state.get("multiplex_state")
                slots = mux.available_slots if mux is not None else self.tracker.multiplex_controller.multiplex_count
                if slots > len(batches[session]):
                    target = session
                    break
            if target is None:
                target = SAM31StreamingSession(self.tracker, **self.options)
                target.start(self.frame_h, self.frame_w)
                self.sessions.append(target)
                batches[target] = []
                target.log.info("toolbox: opened SAM multiplex session %d; existing sessions have no unused slots",
                                len(self.sessions))
            batches[target].append(box)
            self.init_order.append(oid)
        return self._advance(frame, batches, initial_masks)

    def step(self, frame):
        return self._advance(frame, {}) if self.init_order else []

    def _advance(self, frame, batches, initial_masks=None):
        shared_features = None
        for session in self.sessions:
            errors = session.stats["errors"]
            backbone_calls = session.stats["backbone_calls"]
            boxes = batches.get(session, [])
            if boxes:
                options = {}
                if initial_masks is not None:
                    options['initial_masks'] = {int(b[0]): initial_masks[int(b[0])] for b in boxes}
                session.add_objects(frame, boxes, cached_features=shared_features, **options)
            else:
                session.step(frame, cached_features=shared_features)
            self.stats["backbone_calls"] += session.stats["backbone_calls"] - backbone_calls
            if session.stats["errors"] != errors:
                self.stats["errors"] += session.stats["errors"] - errors
                raise SessionError("SAM multiplex session failed; see its error trace")
            shared_features = session.state["cached_features"][session.frame_index]
        self.frame_index += 1
        return self.held_output()

    def remove_objects(self, obj_ids):
        removed = set(int(oid) for oid in obj_ids)
        for session in list(self.sessions):
            local = removed.intersection(session.init_order)
            if local:
                session.remove_objects(local)
            if not session.init_order:
                session.close(run_gc=False)
                self.sessions.remove(session)
                session.log.info("toolbox: released empty multiplex session; %d session(s) remain", len(self.sessions))
        self.init_order = [oid for oid in self.init_order if oid not in removed]

    def multiplex_info(self):
        return {"num_buckets": sum(s.multiplex_info()["num_buckets"] for s in self.sessions),
                "total_valid_entries": len(self.init_order),
                "capacity": self.tracker.multiplex_controller.multiplex_count}
