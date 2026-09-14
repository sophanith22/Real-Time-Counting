# main.py

from typing import Any

import cv2
import os
import time
import random
import csv
import numpy as np
from datetime import datetime
from collections import defaultdict, deque

from tracker.person_tracker import PersonTracker
from utils.zone_crossing import ZoneCrossingDetector
from utils.cropper import crop_person
from feature_extractor.reid_extractor import FeatureExtractor
from memory.feature_memory import FeatureMemory
from decision.decision_logic import DecisionEngine
from config.log import get_logger
from config.settings import (
    get_zone_points, TRACKER_CONFIG, YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    YOLO_PERSON_CLASS_ID, DEVICE, VIDEO_SOURCE, PROCESS_WIDTH, PROCESS_HEIGHT,
    CROP_BUFFER_SIZE, MAX_GALLERY_SIZE, THUMB_SIZE,
    MEMORY_MAX_AGE_SECONDS, MEMORY_MAX_VIEWS,
)

log = get_logger("main")

# --- Setup: build every component one time, at startup ---
tracker = PersonTracker(
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
    person_class_id=YOLO_PERSON_CLASS_ID,
    tracker_config=TRACKER_CONFIG,
    device=DEVICE
)

# --- Zone crossing detector ---
line_detector = ZoneCrossingDetector(
    zone_points=get_zone_points()
)

# 3. Create Re-ID, Memory, and Decision Objects
extractor = FeatureExtractor(device=DEVICE)
memory = FeatureMemory(max_age_seconds=MEMORY_MAX_AGE_SECONDS, max_views=MEMORY_MAX_VIEWS)
decision_engine = DecisionEngine(memory=memory)

#4. Create Folders and Log File
os.makedirs("output/crops", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)

# --- Log file setup ---
session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
log_path = os.path.join("output/logs", session_name)
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "current_track_id", "decision", "matched_memory_index",
        "matched_old_track_id"
    ])


def log_entry(track_id, decision, matched_index, matched_track_id):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(), track_id, decision, matched_index,
            matched_track_id
        ])


log.info(f"Running on device: {DEVICE}")

# --- 6. Open the Camera and Capture ---
def open_capture():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


cap = open_capture()
if cap is None:
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")


def read_frame(cap) -> tuple[bool, Any]:
    """
    Grab the LATEST frame (skipping stale buffered frames) and return it.
    Returns (ok, frame).
    """
    for _ in range(2):
        cap.grab()
    ok, frame = cap.retrieve()
    return ok, frame


# 7. Create Counters and Buffers
prev_time = time.time()
entry_count = 0

# --- This stores recent pictures for every tracking ID. ---
crop_buffer = defaultdict(lambda: deque(maxlen=CROP_BUFFER_SIZE))

# --- Gallery strip setup ---
gallery_crops = []
id_colors = {}

# 8. Person Colors
def get_color(track_id):
    if track_id not in id_colors:
        rng = random.Random(track_id)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        id_colors[track_id] = color
    return id_colors[track_id]


# 9. Build the Gallery
def build_gallery_strip(crops, thumb_size, max_size, strip_width):
    strip = np.zeros((thumb_size, strip_width, 3), dtype=np.uint8)
    x_offset = 10
    for track_id, crop in crops[-max_size:]:
        thumb = cv2.resize(crop, (thumb_size, thumb_size))
        if x_offset + thumb_size > strip_width:
            break
        strip[0:thumb_size, x_offset:x_offset + thumb_size] = thumb
        color = get_color(track_id)
        cv2.putText(strip, f"ID {track_id}", (x_offset + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        x_offset += thumb_size + 10
    return strip


# 10. Average Person Features
def get_averaged_feature(track_id):
    crops = list(crop_buffer[track_id])
    if len(crops) == 0:
        return None
    vectors = [v for v in (extractor.extract(c) for c in crops)
               if v is not None]
    if len(vectors) == 0:
        return None
    averaged = np.mean(vectors, axis=0)
    norm = np.linalg.norm(averaged)
    if norm > 0:
        averaged = averaged / norm
    return averaged


# 11. Largest area crop selection for best quality
def get_best_crop(track_id):
    """
    Pick the BEST quality crop from the buffer - the one with
    the LARGEST area (width x height). Since people often appear
    biggest/clearest just before reaching a far-away door line,
    this usually gives a better photo than just using the very
    last frame, which is often the smallest/farthest moment.
    """
    crops = list(crop_buffer[track_id])
    if len(crops) == 0:
        return None

    best_crop = max(crops, key=lambda c: c.shape[0] * c.shape[1])
    return best_crop


# --- 12. Main loop ---
frame_count = 0

while True:
    ok, frame = read_frame(cap)
    frame_count += 1

    if not ok:
        log.warning(f"Failed to retrieve frame at count {frame_count}; attempting reconnect...")
        cap.release()
        cap = None
        while cap is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            log.info("Trying to reconnect to the camera...")
            cap = open_capture()
            if cap is not None:
                log.info("Camera reconnected.")
                break
            time.sleep(1)

        if cap is None or not cap.isOpened():
            break

        continue

    frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))  # Resize to a smaller size for faster processing
    clean_frame = frame.copy()  # Filter the clean frame without boxes, text, or zone drawings.

    # --- 13. Tracking and Detection & 14. Draw the Door Zone---
    tracks = tracker.track(frame)
    zone_pts: np.ndarray[Any, np.dtype[np.signedinteger]] = np.array(get_zone_points(), np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # --- 15. Process each tracked person ---
    for (track_id, x1, y1, x2, y2, conf) in tracks:
        color = get_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- 16. Crop, Buffer, and Check for Crossing ---
        buffered_crop = crop_person(clean_frame, (x1, y1, x2, y2))
        if buffered_crop is not None:
            crop_buffer[track_id].append(buffered_crop)

    # --- 17. Check Door Entry ---
        crossed = line_detector.check(
            track_id, (x1, y1, x2, y2)
        )

        if crossed:
            # 18. Count the Entry
            entry_count += 1
            log.info(f"Entry Event! Track ID {track_id} crossed the line.")

            # 19. Create an Appearance Feature
            averaged_vector = get_averaged_feature(track_id)
            if averaged_vector is not None:
                # 20. Decision Logic: NEW, DUPLICATE, or COUNT_AGAIN
                decision, matched_index, matched_track_id, _, _ = decision_engine.evaluate(
                    track_id=track_id, feature_vector=averaged_vector
                )
                log.info(f"  Decision: {decision}, matched_old_track_id: {matched_track_id}")
                log_entry(
                    track_id, decision, matched_index, matched_track_id
                )

                # 21. Save the best crop for the gallery
                best_crop = get_best_crop(track_id)
                if best_crop is not None:
                    filename = f"output/crops/person_{track_id}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, best_crop)
                    gallery_crops.append((track_id, best_crop))
            else:
                log.warning(f"  No valid crops for ID {track_id}, skipped.")

            crop_buffer[track_id].clear()

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"Entries: {entry_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if len(gallery_crops) > 0:
        gallery_strip = build_gallery_strip(gallery_crops, THUMB_SIZE, MAX_GALLERY_SIZE, frame.shape[1])
        frame = np.vstack([frame, gallery_strip])

    cv2.imshow("People Counting System - main.py", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Final summary ---
unique_people = memory.count()
with open(log_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([])
    writer.writerow(["SUMMARY"])
    writer.writerow(["total_crossing_events", entry_count])
    writer.writerow(["unique_people_counted", unique_people])

log.info(f"Final Results: {entry_count} crossing events, {unique_people} unique people counted.")
log.info(f"Log saved to: {log_path}")