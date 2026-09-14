# detector/test_detector.py

import cv2
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detector.yolo_detector import PersonDetector
from config.settings import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    YOLO_PERSON_CLASS_ID, DEVICE, VIDEO_SOURCE
)

detector = PersonDetector(
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
    person_class_id=YOLO_PERSON_CLASS_ID,
    device=DEVICE
)

print(f"Running on device: {DEVICE}")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

prev_time = time.time()

while True:
    for _ in range(2):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        break

    # Resize frame - change the video smaller or bigger

    frame = cv2.resize(frame, (1440, 800))  # Resize to 1440x800 for better performance

    detections = detector.detect(frame)

    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"People detected: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Module 1 - Detector Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()