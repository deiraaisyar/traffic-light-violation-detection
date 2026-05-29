from modules.detector import VehicleDetector

detector = VehicleDetector(model_dir="outputs")

# Loop per frame:
detections = detector.detect(frame)
# → [(x, y, w, h, proba), ...]

for x, y, w, h, proba in detections:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)