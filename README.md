# Traffic Light Violation Detection

<video src="video_output/example.mp4" controls width="900"></video>

If the preview does not load in your viewer, open: [video_output/example.mp4](video_output/example.mp4).

## Team

| Name | NIM |
|---|---|
| Deira Aisya Refani | 24/532821/PA/22539 |
| Finanazwa Ayesha | 24/532953/PA/22556 |
| Farrel Tsaqif Anindyo | 24/536735/PA/22769 |
| Yohana Butar Butar | 24/546690/PA/23212 |

## Project Summary

This project detects traffic light violations in video using object detection, tracking, traffic light state detection, and stop-line crossing logic.

## Method (Based on Notebook)

The main pipeline in `notebook.ipynb` uses:

1. YOLOv11 for vehicle detection (car, motorcycle, bus, truck, etc.).
2. SORT (Kalman Filter + Hungarian matching) for multi-object tracking.
3. HSV-based traffic light color detection (RED, YELLOW, GREEN) with temporal smoothing.
4. ROI polygon filtering to focus only on the valid road area.
5. Stop-line crossing check to mark violations when a tracked vehicle crosses during RED.
6. Image enhancement (gamma correction, CLAHE, unsharp masking) before detection.

## Runtime Setup

The notebook was run on Google Colab using GPU T4.

## Output

The output video contains:

- Detected and tracked vehicle IDs.
- Current traffic light state.
- Total violation count.