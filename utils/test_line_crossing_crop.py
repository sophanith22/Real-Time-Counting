# utils/test_line_crossing_crop.py
import csv
import cv2
import sys
import os
import time
import random
import numpy as np
from collections import defaultdict, deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracker.person_tracker import PersonTracker
from utils.line_crossing import LineCrossingDetector
from utils.cropper import crop_person
from feature_extractor.reid_extractor import FeatureExtractor
from memory.feature_memory import FeatureMemory
from decision.decision_logic import DecisionEngine
from config.settings import (
    TRACKER_CONFIG, YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    YOLO_PERSON_CLASS_ID, DEVICE, VIDEO_SOURCE,
    LINE_TYPE, LINE_POSITION, LINE_INSIDE_POSITIVE_SIDE
)

# --- Setup ---
tracker = PersonTracker(
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
    person_class_id=YOLO_PERSON_CLASS_ID,
    tracker_config=TRACKER_CONFIG,
    device=DEVICE
)

line_detector = LineCrossingDetector(
    line_position=LINE_POSITION,
    line_type=LINE_TYPE,
    inside_is_positive_side=LINE_INSIDE_POSITIVE_SIDE
)

extractor = FeatureExtractor(device=DEVICE)
memory = FeatureMemory(max_age_seconds=3600)
decision_engine = DecisionEngine(memory=memory, similarity_threshold=0.85, time_window_seconds=300)

os.makedirs("output/crops", exist_ok=True)

# --- NEW: CSV log setup ---
log_file_path = "output/decision_log.csv"
log_file = open(log_file_path, mode="w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["filename", "track_id", "decision", "matched_index", "timestamp"])

print(f"Running on device: {DEVICE}")

#--- Video Capture Setup ---

## Received data by UDP from the camera.
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

## Received data by TCP from the camera.
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

#cap = cv2.VideoCapture(VIDEO_SOURCE)
#cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#if not cap.isOpened():
    #raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

prev_time = time.time()
entry_count = 0

BUFFER_SIZE = 5
crop_buffer = defaultdict(lambda: deque(maxlen=BUFFER_SIZE))

gallery_crops = []
MAX_GALLERY_SIZE = 5
THUMB_WIDTH = 60
THUMB_HEIGHT = 100 # Size of each thumbnail in the gallery crop strip picture.

id_colors = {}


def get_color(track_id):
    if track_id not in id_colors:
        rng = random.Random(track_id)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        id_colors[track_id] = color
    return id_colors[track_id]


def build_gallery_strip(crops, thumb_width, thumb_height, max_size, strip_width):
    strip = np.zeros((thumb_height, strip_width, 3), dtype=np.uint8)
    x_offset = 10

    for track_id, crop in crops[-max_size:]:
        thumb = cv2.resize(crop, (thumb_width, thumb_height))
        if x_offset + thumb_width > strip_width:
            break

        strip[0:thumb_height, x_offset:x_offset + thumb_width] = thumb

        color = get_color(track_id)
        cv2.putText(strip, f"ID {track_id}", (x_offset + 2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        x_offset += thumb_width + 10

    return strip


def get_averaged_feature(track_id):
    """
    Take all buffered crops for this track_id, extract a feature
    vector from each, then average them into ONE final vector.
    """
    crops = list(crop_buffer[track_id])

    if len(crops) == 0:
        return None

    vectors = []
    for crop in crops:
        vec = extractor.extract(crop)
        if vec is not None:
            vectors.append(vec)

    if len(vectors) == 0:
        return None

    averaged = np.mean(vectors, axis=0)

    norm = np.linalg.norm(averaged)
    if norm > 0:
        averaged = averaged / norm

    return averaged


while True:
    for _ in range(2):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        break

    frame = cv2.resize(frame, (1024, 768)) # Resize to a smaller size for faster processing
    clean_frame = frame.copy()

    tracks = tracker.track(frame)

    if LINE_TYPE == "horizontal":
        cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 255, 255), 2)
    else:
        cv2.line(frame, (LINE_POSITION, 0), (LINE_POSITION, frame.shape[0]), (0, 255, 255), 2)

    for (track_id, x1, y1, x2, y2, conf) in tracks:
        color = get_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        buffered_crop = crop_person(clean_frame, (x1, y1, x2, y2))
        if buffered_crop is not None:
            crop_buffer[track_id].append(buffered_crop)

        crossed = line_detector.check(track_id, (x1, y1, x2, y2))

        if crossed:
            entry_count += 1
            print(f"Entry Event! Track ID {track_id} crossed the line.")

            averaged_vector = get_averaged_feature(track_id)

            if averaged_vector is not None:
                decision, matched_index, matched_track_id, similarity, time_since_seen = decision_engine.evaluate(
                    track_id=track_id, feature_vector=averaged_vector
                )
                print(f"  Decision: {decision}, matched_old_track_id: {matched_track_id}, "
                      f"similarity: {similarity}, time_since_seen: {time_since_seen}, "
                      f"used {len(crop_buffer[track_id])} buffered crops")

                latest_crop = crop_buffer[track_id][-1]
                filename = f"output/crops/person_{track_id}_{int(time.time())}.jpg"
                cv2.imwrite(filename, latest_crop)
                print(f"Saved crop: {filename}")

                # --- NEW: write this decision into the CSV log ---
                log_writer.writerow([filename, track_id, decision, matched_index, time.time()])
                log_file.flush()

                gallery_crops.append((track_id, latest_crop))
            else:
                print(f"Warning: no valid crops buffered for ID {track_id}, skipped.")

            crop_buffer[track_id].clear()

    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"Entries: {entry_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if len(gallery_crops) > 0:
        gallery_strip = build_gallery_strip(
            gallery_crops, THUMB_WIDTH, THUMB_HEIGHT, MAX_GALLERY_SIZE, frame.shape[1] # type: ignore
        )
        frame = np.vstack([frame, gallery_strip])

    cv2.imshow("Module 3+4+5 - Full Pipeline Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()  # NEW: properly close the CSV file