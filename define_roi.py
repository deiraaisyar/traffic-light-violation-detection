"""
define_roi.py  —  Interactive polygon ROI editor
=================================================
Usage:
    python define_roi.py video/input/day.mp4

Controls:
    Left-click      : add a point to the current polygon
    Right-click     : undo last point in current polygon
    Enter / Space   : finish current polygon and start a new one
    D               : delete the last completed polygon
    S               : save all polygons to roi_config.json
    R               : reset — clear everything and start over
    Q / Esc         : quit (will prompt to save if unsaved changes exist)

Output:
    roi_config.json  in the same folder as this script
"""

import cv2
import json
import sys
import numpy as np
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent
OUTPUT_CONFIG  = ROOT / "roi_config.json"
DISPLAY_WIDTH  = 1100          # resize for display only; coords saved at original res

COLOURS = [
    (0,   255, 100),   # green
    (0,   180, 255),   # orange
    (255,  60, 120),   # pink
    (80,  200, 255),   # yellow
    (200,  80, 255),   # purple
    (255, 255,   0),   # cyan
]

FONT       = cv2.FONT_HERSHEY_SIMPLEX
ALPHA      = 0.25               # polygon fill opacity


# ── state ─────────────────────────────────────────────────────────────────────
class ROIEditor:
    def __init__(self, frame: np.ndarray):
        self.orig_h, self.orig_w = frame.shape[:2]
        self.scale  = min(1.0, DISPLAY_WIDTH / self.orig_w)
        dw = int(self.orig_w * self.scale)
        dh = int(self.orig_h * self.scale)
        self.display_size = (dw, dh)
        self.base_frame = cv2.resize(frame, self.display_size)

        self.polygons: list[list[tuple[int,int]]] = []   # completed (orig coords)
        self.current:  list[tuple[int,int]] = []          # in-progress (orig coords)
        self.unsaved   = False
        self.mouse_pos = (0, 0)

    # ── coordinate helpers ────────────────────────────────────────────────────
    def _to_orig(self, x, y):
        return (int(x / self.scale), int(y / self.scale))

    def _to_disp(self, x, y):
        return (int(x * self.scale), int(y * self.scale))

    # ── mouse callback ────────────────────────────────────────────────────────
    def on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append(self._to_orig(x, y))
            self.unsaved = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current:
                self.current.pop()

    # ── actions ───────────────────────────────────────────────────────────────
    def finish_polygon(self):
        if len(self.current) >= 3:
            self.polygons.append(list(self.current))
            print(f"[+] Polygon {len(self.polygons)} saved ({len(self.current)} pts)")
        elif self.current:
            print("[!] Need at least 3 points to close a polygon — ignored.")
        self.current = []

    def delete_last(self):
        if self.polygons:
            self.polygons.pop()
            self.unsaved = True
            print(f"[-] Last polygon deleted. {len(self.polygons)} remaining.")

    def reset(self):
        self.polygons.clear()
        self.current.clear()
        self.unsaved = False
        print("[*] Reset — all polygons cleared.")

    def save(self):
        data = {
            "image_width":  self.orig_w,
            "image_height": self.orig_h,
            "polygons": [
                {"id": i, "points": poly}
                for i, poly in enumerate(self.polygons)
            ],
        }
        OUTPUT_CONFIG.write_text(json.dumps(data, indent=2))
        self.unsaved = False
        print(f"[✓] Saved {len(self.polygons)} polygon(s) → {OUTPUT_CONFIG}")

    # ── rendering ─────────────────────────────────────────────────────────────
    def render(self) -> np.ndarray:
        canvas = self.base_frame.copy()
        overlay = canvas.copy()

        # ── completed polygons ────────────────────────────────────────────
        for i, poly in enumerate(self.polygons):
            colour = COLOURS[i % len(COLOURS)]
            pts_d  = np.array([self._to_disp(*p) for p in poly], dtype=np.int32)
            cv2.fillPoly(overlay, [pts_d], colour)
            cv2.polylines(canvas, [pts_d], isClosed=True, color=colour, thickness=2)
            # label at centroid
            cx = int(np.mean(pts_d[:, 0]))
            cy = int(np.mean(pts_d[:, 1]))
            cv2.putText(canvas, f"ROI {i}", (cx - 20, cy),
                        FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"ROI {i}", (cx - 20, cy),
                        FONT, 0.7, colour,    1, cv2.LINE_AA)
            # vertex dots
            for pt in pts_d:
                cv2.circle(canvas, tuple(pt), 5, colour, -1)
                cv2.circle(canvas, tuple(pt), 5, (0,0,0), 1)

        cv2.addWeighted(overlay, ALPHA, canvas, 1 - ALPHA, 0, canvas)

        # ── in-progress polygon ───────────────────────────────────────────
        if self.current:
            colour = COLOURS[len(self.polygons) % len(COLOURS)]
            pts_d  = [self._to_disp(*p) for p in self.current]
            for pt in pts_d:
                cv2.circle(canvas, pt, 5, colour, -1)
                cv2.circle(canvas, pt, 5, (0,0,0), 1)
            # draw lines between placed points
            for a, b in zip(pts_d, pts_d[1:]):
                cv2.line(canvas, a, b, colour, 2)
            # rubber-band line to current mouse
            cv2.line(canvas, pts_d[-1], self.mouse_pos, colour, 1, cv2.LINE_AA)
            # close preview
            if len(pts_d) >= 2:
                cv2.line(canvas, self.mouse_pos, pts_d[0], colour, 1, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────────────────
        lines = [
            f"Polygons: {len(self.polygons)}  |  Current pts: {len(self.current)}",
            "LClick=add  RClick=undo  Enter=finish  D=del last  S=save  R=reset  Q=quit",
        ]
        if self.unsaved:
            lines[0] += "  [UNSAVED]"

        for li, txt in enumerate(lines):
            y = 22 + li * 22
            cv2.putText(canvas, txt, (8, y), FONT, 0.52, (0,0,0),       2, cv2.LINE_AA)
            cv2.putText(canvas, txt, (8, y), FONT, 0.52, (255,255,255),  1, cv2.LINE_AA)

        return canvas


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── load a representative frame ───────────────────────────────────────
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        # auto-pick first video in video/input/
        exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
        videos = sorted(f for f in (ROOT / "video" / "input").iterdir()
                        if f.suffix.lower() in exts)
        if not videos:
            print("[ERROR] No video found. Pass video path as argument.")
            sys.exit(1)
        video_path = videos[0]

    cap = cv2.VideoCapture(str(video_path))
    # seek to 10 % of video for a more representative frame
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 10))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read frame from video.")
        sys.exit(1)

    print(f"[INFO] Video     : {video_path.name}  ({frame.shape[1]}×{frame.shape[0]})")
    print(f"[INFO] Config out: {OUTPUT_CONFIG}")
    print()

    # ── load existing config if present ───────────────────────────────────
    editor = ROIEditor(frame)
    if OUTPUT_CONFIG.exists():
        try:
            data = json.loads(OUTPUT_CONFIG.read_text())
            for poly in data.get("polygons", []):
                pts = [tuple(p) for p in poly["points"]]
                if len(pts) >= 3:
                    editor.polygons.append(pts)
            print(f"[INFO] Loaded {len(editor.polygons)} existing polygon(s) from {OUTPUT_CONFIG.name}")
        except Exception as e:
            print(f"[WARN] Could not load existing config: {e}")

    # ── window & mouse ────────────────────────────────────────────────────
    win = "ROI Editor  |  S=save  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, *editor.display_size)
    cv2.setMouseCallback(win, editor.on_mouse)

    print("[READY] Click to place points. Press Enter to close a polygon.\n")

    while True:
        cv2.imshow(win, editor.render())
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 32):                        # Enter or Space
            editor.finish_polygon()
        elif key in (ord("d"), ord("D")):
            editor.delete_last()
        elif key in (ord("s"), ord("S")):
            editor.save()
        elif key in (ord("r"), ord("R")):
            editor.reset()
        elif key in (ord("q"), ord("Q"), 27):      # Q or Esc
            if editor.unsaved and editor.polygons:
                print("[?] Unsaved polygons — saving automatically before quit.")
                editor.save()
            break

    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()