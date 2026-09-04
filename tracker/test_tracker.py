# tracker/test_tracker.py

import cv2
import sys
import os
import time
import random # random is used to generate random colors for each track ID, so that each person being tracked can be visually distinguished by a unique color.

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracker.person_tracker import PersonTracker
from config.settings import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    YOLO_PERSON_CLASS_ID, TRACKER_CONFIG, DEVICE, VIDEO_SOURCE
)

tracker = PersonTracker(
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
    person_class_id=YOLO_PERSON_CLASS_ID,
    tracker_config=TRACKER_CONFIG,
    device=DEVICE
)

print(f"Running on device: {DEVICE}")

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

# Give each track_id a consistent color, so it's easy to see
# visually when an ID stays the same vs when it changes.
id_colors = {}

def get_color_for_id(track_id):
    """
    Return a color for this ID. Same ID always gets the same color,
    picked randomly the first time we see that ID.
    """
    if track_id not in id_colors:
        random.seed(track_id)  # same ID = same "random" color every time
        id_colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return id_colors[track_id]

prev_time = time.time()

while True:
    for _ in range(2):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        break

    frame = cv2.resize(frame, (1024, 768))  # Resize frame to a fixed size for consistent processing

    tracks = tracker.track(frame)
    # Draw bounding boxes and track IDs with different colors on the frame.
    for (track_id, x1, y1, x2, y2, conf) in tracks:
        color = get_color_for_id(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"People tracked: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Module 2 - Tracker Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #cap used to capture the video from the source, and we need to release it after we're done to free up resources.
cv2.destroyAllWindows()