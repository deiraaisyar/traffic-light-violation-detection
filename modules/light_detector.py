# traffic_light_detector.py

import cv2
import numpy as np
from collections import deque


class TrafficLightDetector:

    def __init__(self):

        self.rois = [
            {
                "name": "Center Pole",
                "rect": (590, 80, 30, 60)
            }
        ]

        self.hsv_thresholds = {
            "RED": {
                "lower1": np.array([0, 120, 80]),
                "upper1": np.array([10, 255, 255]),
                "lower2": np.array([165, 120, 80]),
                "upper2": np.array([179, 255, 255])
            },

            "YELLOW": {
                "lower": np.array([10, 100, 80]),
                "upper": np.array([35, 255, 255])
            },

            "GREEN": {
                "lower": np.array([36, 60, 60]),
                "upper": np.array([95, 255, 255])
            }
        }

        self.pixel_threshold = 30

        self.history_size = 9
        self.min_confirm_frames = 6

        self.kernel = np.ones((3, 3), np.uint8)

        self.state_histories = {
            roi["name"]: deque(maxlen=self.history_size)
            for roi in self.rois
        }

        self.stable_states = {
            roi["name"]: "OFF"
            for roi in self.rois
        }

    def detect(self, frame):

        detections = []

        for roi in self.rois:

            roi_name = roi["name"]

            x, y, w, h = roi["rect"]

            roi_frame = frame[y:y+h, x:x+w]

            if roi_frame.size == 0:
                continue

            state = self._detect_color(
                roi_frame,
                roi_name
            )

            detections.append({
                "name": roi_name,
                "bbox": (x, y, w, h),
                "state": state
            })

        return detections

    def _detect_color(
        self,
        roi_frame,
        roi_name
    ):

        blurred = cv2.GaussianBlur(
            roi_frame,
            (5, 5),
            0
        )

        hsv = cv2.cvtColor(
            blurred,
            cv2.COLOR_BGR2HSV
        )

        red_mask = self._create_red_mask(hsv)

        yellow_mask = cv2.inRange(
            hsv,
            self.hsv_thresholds["YELLOW"]["lower"],
            self.hsv_thresholds["YELLOW"]["upper"]
        )

        green_mask = cv2.inRange(
            hsv,
            self.hsv_thresholds["GREEN"]["lower"],
            self.hsv_thresholds["GREEN"]["upper"]
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        green_mask = cv2.morphologyEx(
            green_mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        state = "OFF"

        if (
            red_pixels > self.pixel_threshold and
            red_pixels > yellow_pixels and
            red_pixels > green_pixels
        ):
            state = "RED"

        elif (
            yellow_pixels > self.pixel_threshold and
            yellow_pixels > red_pixels and
            yellow_pixels > green_pixels
        ):
            state = "YELLOW"

        elif (
            green_pixels > self.pixel_threshold and
            green_pixels > red_pixels and
            green_pixels > yellow_pixels
        ):
            state = "GREEN"

        return self._temporal_smoothing(
            roi_name,
            state
        )

    def _create_red_mask(self, hsv):

        mask1 = cv2.inRange(
            hsv,
            self.hsv_thresholds["RED"]["lower1"],
            self.hsv_thresholds["RED"]["upper1"]
        )

        mask2 = cv2.inRange(
            hsv,
            self.hsv_thresholds["RED"]["lower2"],
            self.hsv_thresholds["RED"]["upper2"]
        )

        return cv2.bitwise_or(
            mask1,
            mask2
        )

    def _temporal_smoothing(
        self,
        roi_name,
        state
    ):

        history = self.state_histories[roi_name]

        history.append(state)

        if history.count("RED") >= self.min_confirm_frames:
            self.stable_states[roi_name] = "RED"

        elif history.count("YELLOW") >= self.min_confirm_frames:
            self.stable_states[roi_name] = "YELLOW"

        elif history.count("GREEN") >= self.min_confirm_frames:
            self.stable_states[roi_name] = "GREEN"

        elif history.count("OFF") >= self.min_confirm_frames:
            self.stable_states[roi_name] = "OFF"

        return self.stable_states[roi_name]
