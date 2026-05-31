import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=10,      
    qualityLevel=0.2,
    minDistance=5,
    blockSize=7,
)

MAX_MISSED_FRAMES = 10
IOU_THRESHOLD = 0.25
MAX_CENTROID_DRIFT = 80


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]     
    confidence: float

    # Lucas-Kanade state
    pts: Optional[np.ndarray] = None     
    prev_gray: Optional[np.ndarray] = None

    missed: int = 0
    age: int = 0                        
    history: list[tuple[int, int]] = field(default_factory=list)  
    
    @property
    def centroid(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def is_confirmed(self) -> bool:
        return self.age >= 3


def _iou(b1: tuple, b2: tuple) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter
    return inter / union


def _extract_keypoints(gray: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    roi = gray[y: y + h, x: x + w]
    if roi.size == 0:
        return None

    pts = cv2.goodFeaturesToTrack(roi, **FEATURE_PARAMS)
    if pts is None or len(pts) == 0:
        return None

    pts[:, 0, 0] += x
    pts[:, 0, 1] += y
    return pts.astype(np.float32)


def _lk_predict(
    track: "Track",
    cur_gray: np.ndarray,
) -> Optional[tuple[int, int, int, int]]:
    if track.pts is None or track.prev_gray is None:
        return None

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        track.prev_gray, cur_gray, track.pts, None, **LK_PARAMS
    )

    if new_pts is None or status is None:
        return None

    good_new = new_pts[status.flatten() == 1]
    good_old = track.pts[status.flatten() == 1]

    if len(good_new) < 2:
        return None

    dx = float(np.median(good_new[:, 0, 0] - good_old[:, 0, 0]))
    dy = float(np.median(good_new[:, 0, 1] - good_old[:, 0, 1]))

    x, y, w, h = track.bbox
    pred_bbox = (int(x + dx), int(y + dy), w, h)

    track.pts = good_new.reshape(-1, 1, 2).astype(np.float32)

    return pred_bbox

class VehicleTracker:
    def __init__(
        self,
        iou_threshold: float = IOU_THRESHOLD,
        max_missed: int = MAX_MISSED_FRAMES,
    ):
        self.iou_threshold = iou_threshold
        self.max_missed    = max_missed

        self._tracks: list[Track] = []
        self._next_id: int = 0
        self._prev_gray: Optional[np.ndarray] = None
        
    def update(
        self,
        frame: np.ndarray,
        detections: list[tuple[int, int, int, int, float]],
    ) -> list[Track]:
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self._lk_update_all(cur_gray)

        unmatched_dets, unmatched_trks = self._match(detections)

        matched_trk_ids = set(range(len(self._tracks))) - unmatched_trks
        
        for ti in unmatched_trks:
            self._tracks[ti].missed += 1

        for di in unmatched_dets:
            x, y, w, h, conf = detections[di]
            pts = _extract_keypoints(cur_gray, (x, y, w, h))
            new_track = Track(
                track_id=self._next_id,
                bbox=(x, y, w, h),
                confidence=conf,
                pts=pts,
                prev_gray=cur_gray.copy(),
            )
            new_track.history.append(new_track.centroid)
            self._tracks.append(new_track)
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        for t in self._tracks:
            t.prev_gray = cur_gray.copy()
            t.age += 1
            t.history.append(t.centroid)
            if len(t.history) > 60:      
                t.history.pop(0)
            if t.pts is None or len(t.pts) < 3:
                t.pts = _extract_keypoints(cur_gray, t.bbox)

        self._prev_gray = cur_gray
        return list(self._tracks)

    def reset(self):
        """Clear all tracks (e.g. on scene change)."""
        self._tracks.clear()
        self._next_id = 0
        self._prev_gray = None

    @property
    def confirmed_tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.is_confirmed]

    def _lk_update_all(self, cur_gray: np.ndarray):
        """Flow every active track forward by one frame."""
        for t in self._tracks:
            if t.prev_gray is None:
                continue
            pred = _lk_predict(t, cur_gray)
            if pred is not None:
                t.bbox = pred

    def _match(
        self,
        detections: list[tuple[int, int, int, int, float]],
    ) -> tuple[set[int], set[int]]:

        if not self._tracks or not detections:
            return set(range(len(detections))), set(range(len(self._tracks)))

        n_det = len(detections)
        n_trk = len(self._tracks)

        # Build IoU matrix (n_det x n_trk)
        iou_matrix = np.zeros((n_det, n_trk), dtype=np.float32)
        for di, det in enumerate(detections):
            for ti, trk in enumerate(self._tracks):
                iou_matrix[di, ti] = _iou(det[:4], trk.bbox)

        matched_det  = set()
        matched_trk  = set()

        while True:
            if iou_matrix.size == 0:
                break
            flat_idx = np.argmax(iou_matrix)
            di, ti = divmod(int(flat_idx), n_trk)
            if iou_matrix[di, ti] < self.iou_threshold:
                break

            # Update track with the matched detection
            x, y, w, h, conf = detections[di]
            self._tracks[ti].bbox       = (x, y, w, h)
            self._tracks[ti].confidence = conf
            self._tracks[ti].missed     = 0

            matched_det.add(di)
            matched_trk.add(ti)

            # Zero out used row and column
            iou_matrix[di, :] = 0
            iou_matrix[:, ti] = 0

        unmatched_det = set(range(n_det)) - matched_det
        unmatched_trk = set(range(n_trk)) - matched_trk
        return unmatched_det, unmatched_trk