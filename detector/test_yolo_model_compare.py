# detector/test_yolo_model_compare.py

import cv2
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracker.person_tracker import PersonTracker
from config.settings import (
    YOLO_CONFIDENCE_THRESHOLD, YOLO_PERSON_CLASS_ID,
    DEVICE, VIDEO_SOURCE, TRACKER_CONFIG
)

# --- Models we want to compare ---
MODELS_TO_TEST = [
    "models/yolov8n.pt",
    "models/yolov8s.pt",
    "models/yolov8m.pt",
    "models/yolov8l.pt",
    "models/yolo11n.pt",
    "models/yolo11s.pt",
    "models/yolo11m.pt",
    "models/yolo11l.pt",
]

TEST_DURATION_SECONDS = 15  # how long to test each model


def run_test(model_path):
    print(f"\n=== Testing: {model_path} ===")

    tracker = PersonTracker(
        model_path=model_path,
        confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
        person_class_id=YOLO_PERSON_CLASS_ID,
        tracker_config=TRACKER_CONFIG,
        device=DEVICE
    )

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Could not open video source: {VIDEO_SOURCE}")
        return 0.0, 0.0

    frame_count = 0
    total_detections = 0
    start_time = time.time()

    while (time.time() - start_time) < TEST_DURATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video if it ends early
            continue

        frame = cv2.resize(frame, (960, 540))
        tracks = tracker.track(frame)

        frame_count += 1
        total_detections += len(tracks)

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0

    cap.release()

    print(f"  Device: {DEVICE}")
    print(f"  Frames processed: {frame_count}")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average detections per frame: {avg_detections_per_frame:.2f}")

    return avg_fps, avg_detections_per_frame


results = {}
for model_path in MODELS_TO_TEST:
    fps, detections = run_test(model_path)
    results[model_path] = (fps, detections)

print("\n=== COMPARISON SUMMARY ===")
for model_path, (fps, detections) in results.items():
    print(f"{model_path}: {fps:.2f} FPS, {detections:.2f} avg detections/frame")