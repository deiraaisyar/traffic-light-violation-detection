import cv2
import sys
import time
import random
from pathlib import Path

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.detector import VehicleDetector   # noqa: E402
from modules.tracker  import VehicleTracker    # noqa: E402

# ── config ────────────────────────────────────────────────────────────────────
MODEL_DIR        = ROOT / "model"
VIDEO_INPUT_DIR  = ROOT / "video" / "input"
VIDEO_OUTPUT_DIR = ROOT / "video" / "output"

# detector
CONF_THRESHOLD = 0.6
IOU_THRESHOLD  = 0.3
MIN_CONTOUR    = 500
PADDING        = 20

# tracker
TRACKER_IOU_THRESHOLD = 0.25
TRACKER_MAX_MISSED    = 10

# display
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.55
FONT_THICK    = 1
BOX_THICKNESS = 2
WINDOW_NAME   = "Vehicle Detection + Tracking  |  Q to quit"
DISPLAY_WIDTH = 960


# ── colour palette (one stable colour per track ID) ──────────────────────────
_PALETTE: dict[int, tuple[int, int, int]] = {}

def _track_colour(track_id: int) -> tuple[int, int, int]:
    if track_id not in _PALETTE:
        rng = random.Random(track_id * 2654435761)   # deterministic per ID
        _PALETTE[track_id] = (rng.randint(80, 255), rng.randint(80, 255), rng.randint(80, 255))
    return _PALETTE[track_id]


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_tracks(frame, tracks):
    """Draw bounding box, ID label, and centroid trail for every active track."""
    for t in tracks:
        color = _track_colour(t.track_id)
        x, y, w, h = t.bbox

        # bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, BOX_THICKNESS)

        # label: "ID 3  0.82"
        label = f"ID {t.track_id}  {t.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 4),
                    FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)

        # centroid trail
        pts = t.history
        for i in range(1, len(pts)):
            thickness = max(1, int(3 * i / len(pts)))   # thicker toward current
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    return frame


def draw_overlay(frame, frame_idx, total_frames, n_tracks, fps):
    """Top-left status bar."""
    info = (
        f"Frame {frame_idx + 1}"
        + (f"/{total_frames}" if total_frames else "")
        + f"  |  Tracks: {n_tracks}  |  {fps:.1f} fps"
    )
    # white outline + black fill for readability on any background
    cv2.putText(frame, info, (10, 24), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, info, (10, 24), FONT, 0.6, (0, 0, 0),       1, cv2.LINE_AA)
    return frame


def resize_for_display(frame, max_w):
    if max_w <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    return cv2.resize(frame, (max_w, int(h * max_w / w)))


def get_video_writer(cap, out_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))


# ── main ──────────────────────────────────────────────────────────────────────
def run(video_path: Path):
    print(f"[INFO] Loading model from {MODEL_DIR} ...")
    detector = VehicleDetector(model_dir=str(MODEL_DIR))
    tracker  = VehicleTracker(
        iou_threshold=TRACKER_IOU_THRESHOLD,
        max_missed=TRACKER_MAX_MISSED,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_OUTPUT_DIR / ("tracked_" + video_path.name)
    writer   = get_video_writer(cap, out_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] Video  : {video_path.name}")
    print(f"[INFO] Output : {out_path}")
    print(f"[INFO] Frames : {total_frames if total_frames else '?'}")
    print("[INFO] Press  Q  to quit\n")

    # ── GUI availability check ────────────────────────────────────────────
    gui_available = False
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        gui_available = True
    except cv2.error:
        print("[WARN] OpenCV GUI not available (headless). Output saved to disk only.")
        print("[WARN] To enable live preview:  pip install opencv-python\n")

    frame_idx = 0
    t_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. detect ─────────────────────────────────────────────────────
        detections = detector.detect(
            frame,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            min_contour_area=MIN_CONTOUR,
            padding=PADDING,
        )

        # ── 2. track ──────────────────────────────────────────────────────
        tracks = tracker.update(frame, detections)

        # ── 3. draw ───────────────────────────────────────────────────────
        frame_out = frame.copy()
        draw_tracks(frame_out, tracker.confirmed_tracks)

        elapsed = time.time() - t_start
        fps_cur = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        draw_overlay(frame_out, frame_idx, total_frames,
                     len(tracker.confirmed_tracks), fps_cur)

        # ── 4. save ───────────────────────────────────────────────────────
        writer.write(frame_out)

        # ── 5. display / progress ─────────────────────────────────────────
        if gui_available:
            cv2.imshow(WINDOW_NAME, resize_for_display(frame_out, DISPLAY_WIDTH))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[INFO] Stopped by user.")
                break
        else:
            if frame_idx % 50 == 0:
                pct = f"{100 * frame_idx / total_frames:.1f}%" if total_frames else f"{frame_idx} frames"
                print(f"  [{pct}]  tracks={len(tracker.confirmed_tracks)}  {fps_cur:.1f} fps", end="\r")

        frame_idx += 1

    cap.release()
    writer.release()
    if gui_available:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\n[INFO] Done — {frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} fps avg)")
    print(f"[INFO] Saved to: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────
def find_video() -> Path:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)
        return p

    exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    videos = sorted(f for f in VIDEO_INPUT_DIR.iterdir() if f.suffix.lower() in exts)
    if not videos:
        print(f"[ERROR] No video files found in {VIDEO_INPUT_DIR}")
        sys.exit(1)
    if len(videos) > 1:
        print("[INFO] Multiple videos found:")
        for i, v in enumerate(videos):
            print(f"  [{i}] {v.name}")
        idx = input("Select number [0]: ").strip()
        return videos[int(idx) if idx.isdigit() else 0]
    return videos[0]


if __name__ == "__main__":
    run(find_video())