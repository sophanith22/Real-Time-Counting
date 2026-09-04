# People Counting System Explanation

## 1. What Is This System?

This project counts people who enter a door area.

The system uses:

- A camera or an RTSP camera stream
- YOLO to find people
- ByteTrack to give each person an ID
- A polygon zone to detect entry
- OSNet Re-ID to compare person appearance
- Memory to store people already counted
- CSV files to save results

The main file is `main.py`.

## 2. Main System Flow

```text
Start the program
    |
Load the models and settings
    |
Open the camera
    |
Read one video frame
    |
Find and track people
    |
Draw boxes and the door zone
    |
Save recent person crops
    |
Check if a person enters the zone
    |
Create an appearance feature
    |
Compare the feature with memory
    |
Return NEW, DUPLICATE, or COUNT_AGAIN
    |
Save the result in a CSV file
    |
Show the video
    |
Repeat until the user presses q
    |
Close the camera and save the summary
```

## 3. Imports

```python
from typing import Any
```

`Any` is used for a NumPy type description later in the file.

```python
import cv2
import sys
import os
import time
import random
import csv
import numpy as np
```

These libraries provide common tools:

- `cv2` reads video, draws shapes, saves pictures, and shows windows.
- `sys` is imported, but it is not used in this file.
- `os` creates folders and file paths.
- `time` reads the current time and calculates FPS.
- `random` creates colors for person IDs.
- `csv` writes CSV files.
- `numpy` works with images and number arrays.

```python
from datetime import datetime
from collections import defaultdict, deque
```

- `datetime` creates dates and times.
- `defaultdict` creates a new empty value when needed.
- `deque` stores a limited number of recent images.

```python
from tracker.person_tracker import PersonTracker
from utils.zone_crossing import ZoneCrossingDetector
from utils.cropper import crop_person
from feature_extractor.reid_extractor import FeatureExtractor
from memory.feature_memory import FeatureMemory
from decision.decision_logic import DecisionEngine
```

These are project classes and functions:

- `PersonTracker` finds and tracks people.
- `ZoneCrossingDetector` checks entry into the door zone.
- `crop_person` cuts a person image from a frame.
- `FeatureExtractor` changes a person image into numbers.
- `FeatureMemory` stores old feature vectors.
- `DecisionEngine` decides if a person is new or already counted.

```python
from config.settings import (
    DOOR_ZONE_POINTS, TRACKER_CONFIG, YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD, YOLO_PERSON_CLASS_ID,
    DEVICE, VIDEO_SOURCE
)
```

These values come from `config/settings.py`.
They define the model, camera, zone, confidence, and device.

## 4. Create the Person Tracker

```python
tracker = PersonTracker(
    model_path=YOLO_MODEL_PATH,
    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
    person_class_id=YOLO_PERSON_CLASS_ID,
    tracker_config=TRACKER_CONFIG,
    device=DEVICE
)
```

This creates the tracker one time.

The tracker loads YOLO and ByteTrack.
It finds people and gives each person a tracking ID.

Example result:

```python
(12, 100, 80, 300, 600, 0.91)
```

This means:

- ID is `12`.
- The box is from `(100, 80)` to `(300, 600)`.
- Detection confidence is `0.91`.

The tracker uses `persist=True` inside `person_tracker.py`.
This helps keep the same ID from one frame to the next.

## 5. Create the Door Zone Detector

```python
line_detector = ZoneCrossingDetector(
    zone_points=DOOR_ZONE_POINTS
)
```

The variable is called `line_detector`, but it checks a polygon zone.
The zone is made from four points in `settings.py`.

The detector uses the bottom-center of the person box.
This point is near the person's feet.

An entry happens when the person moves like this:

```text
Outside the zone -> Inside the zone
```

The system checks entry only.
It does not count a person who moves from inside to outside.

### 5.1 Blue Box, Green Block, and Yellow Zone

The blue box and green block in the explanation are only visual labels used to explain the idea.
They are not separate objects in the code.

The real system uses only the yellow polygon zone.
This yellow zone is the actual counting boundary.

The logic is:

```text
Outside the yellow zone -> Cross the yellow zone -> Inside area
```

That means the system does not count just because a person is anywhere in the frame.
It counts only when the person's feet point moves from outside the polygon to inside it.

This matches the real requirement of the project:

- Blue area = outside the home or outside the door
- Yellow polygon = the real door zone or counting line
- Green area = inside the home or inside the final area
- The important event is the crossing from outside to inside

So the drawing is a simple explanation of the same logic.

### 5.2 Why the camera sees front, back, or side views

The camera captures whatever side of the person is facing the camera.
A person can be seen from:

- the front
- the back
- the side

This is normal and does not change the counting zone.

The counting rule is based on location, not face direction.
The system checks where the person is in the frame and whether they crossed the yellow zone.

So the same person can be counted even when:

- one crop is from the back
- another crop is from the front
- another crop is from the side

The counting event is still the same if the feet cross the door boundary.

The Re-ID model is used to recognize that it is the same person even if the appearance changes a little from different angles.

## 6. Create Re-ID, Memory, and Decision Objects

```python
extractor = FeatureExtractor(device=DEVICE)
```

This loads the OSNet Re-ID model.
It changes a person picture into a feature vector.
A feature vector is a list of numbers about the person's appearance.

```python
memory = FeatureMemory(max_age_seconds=3600)
```

This stores feature vectors for one hour.
`3600` seconds is one hour.

```python
decision_engine = DecisionEngine(
    memory=memory,
    similarity_threshold=0.75,
    time_window_seconds=300
)
```

The decision engine compares a new feature with memory.

- Similarity must be at least `0.75` to match.
- A repeat within `300` seconds is too soon.
- `300` seconds is five minutes.

The decisions are:

- `NEW`: the person does not match memory.
- `DUPLICATE`: the person matches and returns too soon.
- `COUNT_AGAIN`: the person matches but returns after five minutes.

## 7. Create Output Folders and a Log File

```python
os.makedirs("output/crops", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)
```

These commands create folders for:

- Saved person images
- CSV log files

`exist_ok=True` means the program does not fail if the folders already exist.

```python
session_name = datetime.now().strftime(
    "session_%Y%m%d_%H%M%S.csv"
)
```

This creates a file name with the current date and time.

Example:

```text
session_20260824_143500.csv
```

```python
log_path = os.path.join("output/logs", session_name)
```

This creates the full path to the log file.

```python
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "current_track_id", "decision", "matched_memory_index",
        "matched_old_track_id"
    ])
```

The new CSV file is opened.
The first row writes the column names.

## 8. The `log_entry` Function

```python
def log_entry(track_id, decision, matched_index, matched_track_id):
```

This function saves one decision in the log.

```python
with open(log_path, "a", newline="") as f:
```

The file is opened in append mode.
Old rows are kept.

```python
writer.writerow([
    datetime.now().isoformat(),
    track_id,
    decision,
    matched_index,
    matched_track_id
])
```

This writes the time, current ID, decision, and old matching ID.
Similarity and time are used internally, but they are not shown in the CSV.

## 9. Open the Camera

```python
print(f"Running on device: {DEVICE}")
```

This prints `cpu` or `cuda`.

```python
cap = cv2.VideoCapture(VIDEO_SOURCE)
```

This opens the video source.
The source can be a webcam, video file, or RTSP stream.

```python
if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video source: {VIDEO_SOURCE}"
    )
```

If the camera cannot open, the program stops with an error.

## 10. Create Counters and Buffers

```python
prev_time = time.time()
entry_count = 0
```

- `prev_time` is used to calculate FPS.
- `entry_count` stores the number of crossing events.

```python
BUFFER_SIZE = 5
crop_buffer = defaultdict(lambda: deque(maxlen=BUFFER_SIZE))
```

Each tracking ID has a buffer with five recent person pictures.
When a new picture arrives, the oldest picture is removed.

```python
gallery_crops = []
MAX_GALLERY_SIZE = 5
THUMB_SIZE = 85
id_colors = {}
```

These values control the saved image gallery.
The gallery shows up to five person thumbnails.

## 11. Create a Color for Each Person

```python
def get_color(track_id):
```

This function returns a color for a person ID.

```python
if track_id not in id_colors:
```

If the ID has no color, the program creates one.

```python
rng = random.Random(track_id)
```

The ID is used as the random seed.
This makes the same ID keep the same color.

```python
color = (
    rng.randint(0, 255),
    rng.randint(0, 255),
    rng.randint(0, 255)
)
```

This creates a random OpenCV BGR color.

```python
id_colors[track_id] = color
return id_colors[track_id]
```

The color is saved and returned.

## 12. Build the Gallery Strip

```python
def build_gallery_strip(crops, thumb_size, max_size, strip_width):
```

This function creates a small image strip.

```python
strip = np.zeros(
    (thumb_size, strip_width, 3),
    dtype=np.uint8
)
```

This creates a black image for the gallery.

```python
x_offset = 10
```

The first thumbnail starts at position 10.

```python
for track_id, crop in crops[-max_size:]:
```

Only the last five saved crops are used.

```python
thumb = cv2.resize(crop, (thumb_size, thumb_size))
```

The person image becomes a small square thumbnail.

```python
strip[0:thumb_size,
      x_offset:x_offset + thumb_size] = thumb
```

The thumbnail is placed in the gallery.

```python
cv2.putText(strip, f"ID {track_id}", ...)
```

The person ID is written on the thumbnail.

```python
return strip
```

The finished gallery is returned.

## 13. Create an Average Feature

```python
def get_averaged_feature(track_id):
```

This function creates one feature from several pictures.

```python
crops = list(crop_buffer[track_id])
```

It gets the recent pictures for the person.

```python
if len(crops) == 0:
    return None
```

If there are no pictures, it returns `None`.

```python
vectors = [
    v for v in (extractor.extract(c) for c in crops)
    if v is not None
]
```

Each picture goes to the Re-ID model.
Invalid results are removed.

```python
if len(vectors) == 0:
    return None
```

If all pictures fail, there is no feature.

```python
averaged = np.mean(vectors, axis=0)
```

The feature vectors are averaged.
This can be more stable than using one picture.

```python
norm = np.linalg.norm(averaged)
if norm > 0:
    averaged = averaged / norm
```

The vector is normalized.

```python
return averaged
```

The final feature vector is returned.

## 14. Select the Best Crop

```python
def get_best_crop(track_id):
```

This function selects the best recent picture.

```python
crops = list(crop_buffer[track_id])
```

It gets the pictures in the person's buffer.

```python
if len(crops) == 0:
    return None
```

If there are no pictures, it returns `None`.

```python
best_crop = max(
    crops,
    key=lambda c: c.shape[0] * c.shape[1]
)
```

The picture with the largest area is selected.
A larger picture usually has more detail.

```python
return best_crop
```

The best picture is returned.

## 15. Start the Main Loop

```python
frame_count = 0

while True:
```

The program starts an endless loop.
It stops when the video ends, the camera fails, or the user presses `q`.

```python
for _ in range(2):
    grabbed = cap.grab()
```

The program grabs two frames.
It later processes the newest grabbed frame.
This can reduce the amount of work.

```python
ret, frame = cap.retrieve()
frame_count += 1
```

The latest frame is loaded and the frame counter increases.

```python
if not ret:
    print(...)
    break
```

If the frame cannot be read, the loop stops.

```python
frame = cv2.resize(frame, (1280, 720))
```

The frame is changed to `1280 x 720`.
This gives a fixed size and can improve speed.

```python
clean_frame = frame.copy()
```

A clean copy is made before drawing boxes and text.
This copy is used for saved person images.

## 16. Detect and Track People

```python
tracks = tracker.track(frame)
```

YOLO finds people and ByteTrack follows them.
The result contains IDs, boxes, and confidence values.

```python
zone_pts = np.array(
    DOOR_ZONE_POINTS,
    np.int32
).reshape((-1, 1, 2))
```

The door points are changed to the format needed by OpenCV.

```python
cv2.polylines(
    frame,
    [zone_pts],
    isClosed=True,
    color=(0, 255, 255),
    thickness=2
)
```

The door polygon is drawn in yellow.

## 17. Draw Each Person

```python
for (track_id, x1, y1, x2, y2, conf) in tracks:
```

The program processes each tracked person.

```python
color = get_color(track_id)
```

The person gets a stable color.

```python
cv2.rectangle(
    frame,
    (x1, y1),
    (x2, y2),
    color,
    2
)
```

A box is drawn around the person.

```python
cv2.putText(frame, f"ID {track_id}", ...)
```

The tracking ID is drawn above the box.

## 18. Crop and Buffer Person Pictures

```python
buffered_crop = crop_person(
    clean_frame,
    (x1, y1, x2, y2)
)
```

The person is cut from the clean frame.
The crop has a small padding area.

The cropper rejects invalid or very small pictures.

```python
if buffered_crop is not None:
    crop_buffer[track_id].append(buffered_crop)
```

A valid picture is added to the person's five-picture buffer.

## 19. Check Door Entry

```python
crossed = line_detector.check(
    track_id,
    (x1, y1, x2, y2)
)
```

The zone detector checks the person's feet point.
It compares the current position with the previous position.

It returns `True` only when the person changes from outside to inside.

## 20. Process a Crossing Event

```python
if crossed:
    entry_count += 1
```

The crossing event counter increases.

```python
print(
    f"Entry Event! Track ID {track_id} crossed the line."
)
```

The event is printed in the terminal.

```python
averaged_vector = get_averaged_feature(track_id)
```

The recent person pictures are changed into one average feature vector.

```python
if averaged_vector is not None:
```

The decision process continues only when a valid feature exists.

```python
decision, matched_index, matched_track_id, _, _ = decision_engine.evaluate(
    track_id=track_id,
    feature_vector=averaged_vector
)
```

The new feature is compared with memory.
The result is `NEW`, `DUPLICATE`, or `COUNT_AGAIN`.

```python
print(
    f"Decision: {decision}, matched_old_track_id: {matched_track_id}"
)
```

The decision is printed.

```python
log_entry(
    track_id, decision, matched_index, matched_track_id
)
```

The decision is saved in the CSV file.

## 21. Save the Best Person Picture

```python
best_crop = get_best_crop(track_id)
```

The largest recent person picture is selected.

```python
filename = (
    f"output/crops/person_{track_id}_"
    f"{int(time.time())}.jpg"
)
```

A file name is created using the track ID and current time.

```python
cv2.imwrite(filename, best_crop)
```

The picture is saved as a JPG file.

```python
gallery_crops.append((track_id, best_crop))
```

The picture is added to the gallery.

```python
crop_buffer[track_id].clear()
```

The old pictures for this person are deleted after the crossing.

If there is no valid feature, the program prints a warning and skips Re-ID.

## 22. Calculate FPS and Draw Information

```python
current_time = time.time()
fps = 1.0 / (current_time - prev_time)
prev_time = current_time
```

The program calculates processed frames per second.

```python
cv2.putText(frame, f"Entries: {entry_count}", ...)
cv2.putText(frame, f"FPS: {fps:.1f}", ...)
```

The entry count and FPS are drawn on the video.

## 23. Show the Gallery

```python
if len(gallery_crops) > 0:
```

If at least one crop was saved, the gallery is created.

```python
gallery_strip = build_gallery_strip(
    gallery_crops,
    THUMB_SIZE,
    MAX_GALLERY_SIZE,
    frame.shape[1]
)
```

This creates the gallery image.

```python
frame = np.vstack([frame, gallery_strip])
```

The gallery is placed below the video frame.

## 24. Show the Video and Read the Keyboard

```python
cv2.imshow(
    "People Counting System - main.py",
    frame
)
```

The current frame is shown in an OpenCV window.

```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

The loop stops when the user presses the `q` key.

## 25. Close the Camera and Windows

```python
cap.release()
cv2.destroyAllWindows()
```

The video source is released.
All OpenCV windows are closed.

## 26. Write the Final Summary

```python
unique_people = memory.count()
```

This counts the feature records currently in memory.

```python
writer.writerow([])
writer.writerow(["SUMMARY"])
writer.writerow(["total_crossing_events", entry_count])
writer.writerow(["unique_people_counted", unique_people])
```

The final summary is added to the CSV log.

```python
print(
    f"Final Results: {entry_count} crossing events, "
    f"{unique_people} unique people counted."
)
```

The final numbers are printed in the terminal.

```python
print(f"Log saved to: {log_path}")
```

The location of the log file is printed.

## 27. Important Difference Between the Counters

`entry_count` counts every zone-crossing event.

`unique_people` counts records stored in Re-ID memory.

These numbers can be different.
For example, one person can enter twice:

- Crossing events: `2`
- Unique memory records: usually `1`

A person who returns after five minutes can get the decision `COUNT_AGAIN`, but the current code does not increase `entry_count` again inside the decision code. The crossing event counter is increased before the Re-ID decision.
