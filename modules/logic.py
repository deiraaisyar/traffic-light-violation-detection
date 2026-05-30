"""Rules / logic engine for traffic-light violations.

This module provides `TrafficViolationLogic`, a small rules engine
that consumes tracked vehicle objects and traffic-light state and
returns structured violations. It intentionally does not handle
drawing or I/O so callers can decide how to record or show results.

Usage (high level):
  - Use `VehicleDetector` + a `tracker` to obtain stable IDs per vehicle.
  - Call `logic.update(tracked_objects, light_state, timestamp)` each frame.

tracked_objects expected format: list of `(id, x, y, w, h)`
light_state expected: one of 'red', 'yellow', 'green'
"""

from typing import List, Tuple, Dict
import time

class TrafficViolationLogic:
    def __init__(self, line_points: Tuple[Tuple[int,int], Tuple[int,int]] | None = None):
        """
        line_points: two endpoints ((x1,y1),(x2,y2)) for the stop line.
        """
        # determine stop-line config
        cfg = None
        if line_points is not None:
            cfg = {"type": "line", "points": [list(line_points[0]), list(line_points[1])]}

        self.mode = None
        self.line_a = None
        self.line_b = None
        if cfg is not None and cfg.get("type") == "line":
            self.mode = "line"
            a, b = cfg["points"]
            self.line_a = (int(a[0]), int(a[1]))
            self.line_b = (int(b[0]), int(b[1]))

        self.vehicles: Dict[int, Dict] = {}  # vid -> {'last_center':(x,y), 'crossed':bool}

    def _center(self, x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        return x + w // 2, y + h // 2

    @staticmethod
    def _side(p: Tuple[int,int], a: Tuple[int,int], b: Tuple[int,int]) -> int:
        # cross product sign to determine which side of line p lies on
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    def update(
        self,
        tracked_objects: List[Tuple[int, int, int, int, int]],
        light_state: str,
        timestamp: float | None = None,
    ) -> List[Dict]:
        """
        Process a frame's tracked objects and return violations.

        Returns list of violations with keys: 'id', 'bbox', 'reason', 'time'
        """
        if timestamp is None:
            timestamp = time.time()

        violations: List[Dict] = []
        for obj in tracked_objects:
            vid, x, y, w, h = obj
            cx, cy = self._center(x, y, w, h)
            prev = self.vehicles.get(vid)

            if prev is None:
                # first time seeing this vehicle
                self.vehicles[vid] = {"last_center": (cx, cy), "crossed": False}
                continue

            prev_cx, prev_cy = prev["last_center"]
            crossed = prev.get("crossed", False)

            if not crossed:
                if self.mode == "line" and self.line_a is not None and self.line_b is not None:
                    # detect side change (crossing) using cross product sign
                    side_prev = self._side((prev_cx, prev_cy), self.line_a, self.line_b)
                    side_curr = self._side((cx, cy), self.line_a, self.line_b)
                    if side_prev * side_curr < 0:
                        if light_state == "red":
                            violations.append({"id": vid, "bbox": (x, y, w, h), "reason": "crossed_on_red", "time": timestamp})
                            crossed = True
                # Note: if no line is configured (mode != 'line'), no crossing check is performed

            # update state for next frame
            prev["last_center"] = (cx, cy)
            prev["crossed"] = crossed

        return violations


# Backwards-compatible convenience: if downstream modules imported `detector`
# from here previously, attempt a soft import. Wrap in try/except to avoid
# crashing if model files are not present in the environment.
try:
    from modules.detector import VehicleDetector  # type: ignore

    try:
        detector = VehicleDetector(model_dir="model")
    except Exception:
        # if loading fails, keep `detector` undefined rather than raising
        detector = None
except Exception:
    detector = None
