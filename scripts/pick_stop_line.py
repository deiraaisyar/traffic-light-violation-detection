import cv2
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Pick two points on a frame to define the stop line and save to JSON.")
parser.add_argument("--video", default=0, help="Path to video file or camera index (default=0)")
parser.add_argument("--out", default="stop_line.json", help="Output JSON file path")
parser.add_argument("--frame", type=int, default=0, help="Frame index to grab for picking (default=0)")
args = parser.parse_args()

points = []


def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((int(x), int(y)))


cap = cv2.VideoCapture(str(args.video)) if not str(args.video).isdigit() else cv2.VideoCapture(int(args.video))
if not cap.isOpened():
    raise SystemExit(f"Cannot open video/camera: {args.video}")

# seek to frame
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ret, frame = cap.read()
if not ret:
    raise SystemExit("Can't read frame from video")

cv2.namedWindow("pick")
cv2.setMouseCallback("pick", on_mouse)

while True:
    display = frame.copy()
    for p in points:
        cv2.circle(display, p, 6, (0, 255, 0), -1)
    if len(points) >= 2:
        cv2.line(display, points[0], points[1], (0, 0, 255), 2)
    cv2.imshow("pick", display)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break
    if len(points) >= 2:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stop_line": {"type": "line", "points": [list(points[0]), list(points[1])]}}
        outp.write_text(json.dumps(payload, indent=2))
        print(f"Saved stop line to {outp}")
        break

cap.release()
cv2.destroyAllWindows()