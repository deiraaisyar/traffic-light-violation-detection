import cv2
import joblib
import numpy as np
from pathlib import Path
from skimage.feature import hog


class VehicleDetector:
    def __init__(self, model_dir: str = "outputs"):
        model_dir = Path(model_dir)

        self.model      = joblib.load(model_dir / "svm_model.joblib")
        self.scaler     = joblib.load(model_dir / "scaler.joblib")
        self.hog_params = joblib.load(model_dir / "hog_config.joblib")

        win = self.hog_params["window_size"]   
        self.WIN_W = win[0]
        self.WIN_H = win[1]

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )


    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.6,
        iou_threshold: float  = 0.3,
        min_contour_area: int = 500,
        padding: int          = 20,
    ) -> list[tuple[int, int, int, int, float]]:

        h_frame, w_frame = frame.shape[:2]


        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self._morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._morph_kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_contour_area:
                continue

            cx, cy, cw, ch = cv2.boundingRect(cnt)
            x1 = max(cx - padding, 0)
            y1 = max(cy - padding, 0)
            x2 = min(cx + cw + padding, w_frame)
            y2 = min(cy + ch + padding, h_frame)

            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            label, proba = self._predict_patch(patch)
            if label == 1 and proba >= conf_threshold:
                detections.append((x1, y1, x2 - x1, y2 - y1, proba))

        return self._nms(detections, iou_threshold)

    def reset_background(self):
        """Reset MOG2 — panggil saat ganti video/scene baru."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )

    def _extract_hog(self, gray_patch: np.ndarray) -> np.ndarray:
        resized = cv2.resize(gray_patch, (self.WIN_W, self.WIN_H))
        return hog(
            resized,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"],
            cells_per_block=self.hog_params["cells_per_block"],
            visualize=False,
            feature_vector=True,
        )

    def _predict_patch(
        self, patch_bgr: np.ndarray
    ) -> tuple[int, float]:
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        feat = self._extract_hog(gray)
        feat_scaled = self.scaler.transform([feat])
        label = int(self.model.predict(feat_scaled)[0])
        proba = float(self.model.predict_proba(feat_scaled)[0][1])
        return label, proba

    @staticmethod
    def _nms(
        detections: list[tuple[int, int, int, int, float]],
        iou_threshold: float = 0.3,
    ) -> list[tuple[int, int, int, int, float]]:
        if not detections:
            return []
        boxes = np.array(
            [[x, y, x + w, y + h, p] for x, y, w, h, p in detections]
        )
        x1, y1, x2, y2, scores = (
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        )
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        kept  = []
        while order.size > 0:
            i = order[0]
            kept.append(i)
            xx1   = np.maximum(x1[i], x1[order[1:]])
            yy1   = np.maximum(y1[i], y1[order[1:]])
            xx2   = np.minimum(x2[i], x2[order[1:]])
            yy2   = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou < iou_threshold]
        return [detections[i] for i in kept]