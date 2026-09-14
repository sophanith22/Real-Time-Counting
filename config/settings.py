# config/settings.py

import os
from pathlib import Path
from typing import List, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # .env support is optional
    def load_dotenv() -> bool:
        return False

# Load .env from the project root (one directory up from this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Device ---
import torch

# Check one time: do we have a GPU(CUDA) available?
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"  # Force CPU for testing, change to "cuda" for GPU if available

# --- Processing resolution ---
# Every frame is resized to this size before detection/tracking.
# IMPORTANT: the door zone polygon must be drawn/calibrated against
# the SAME resolution. See ZONE_REFERENCE_RESOLUTION below.
PROCESS_WIDTH = 1920
PROCESS_HEIGHT = 1200

# --- Path to the YOLO model file ---
"""
YOLO_MODEL_PATH_NANNO = "models/yolov8n.pt" # Nano model, fastest but least accurate. Good for real-time use, but may miss detections.
YOLO_MODEL_PATH_SMALL = "models/yolov8s.pt" # Small model, good for real-time use, but less accurate than medium model.
YOLO_MODEL_PATH_MEDIUM = "models/yolov8m.pt" # Medium model, better accuracy but slower than small model. Not good for real-time use.
YOLO_MODEL_PATH_LARGE = "models/yolov8l.pt" # Large model, best accuracy but slowest. Not good for real-time use.
"""
YOLO_MODEL_PATH_NANNO = "models/yolo11n.pt" # Nano model, fastest but least accurate. Good for real-time use, but may miss detections.
YOLO_MODEL_PATH_SMALL = "models/yolo11s.pt" # Small model, good for real-time use, but less accurate than medium model.
YOLO_MODEL_PATH_MEDIUM = "models/yolo11m.pt" # Medium model, better accuracy but slower than small model. Not good for real-time use.
YOLO_MODEL_PATH_LARGE = "models/yolo11l.pt" # Large model, best accuracy but slowest. Not good for real-time use.

# Prefer the environment override (handy when switching models per camera).
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", YOLO_MODEL_PATH_SMALL)

# Path to the Re-ID model file (OSNet) is specific re-ID of people, not generic Imagenet.
# REID_MODEL_PATH = "models/OsNetReID/osnet_x1_0_msmt17.pth" # is bigger model of OSNet, accuracy is better but slower. Not good for real-time use.
REID_MODEL_PATH = os.getenv("REID_MODEL_PATH", "models/OsNetReID/osnet_x0_25_msmt17.pth") # is smallest model of OSNet, accuracy is lower but faster. Good for real-time use.

# Minimum score to keep a detection (0.0 - 1.0)
YOLO_CONFIDENCE_THRESHOLD = 0.5  # 0.5 is a good balance between false positives and false negatives.

# Class ID for "person" in the YOLO model (usually 0 for COCO dataset)
YOLO_PERSON_CLASS_ID = 0

# Video source: RTSP stream URL, local file path, or 0 for webcam.
# Set it in .env so credentials never end up in source control.
# Default keeps the historical value so the system still runs without a .env.
VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    "rtsp://admin:Hik123456@192.168.88.125:554/ch1/main/av_stream",
)

# Tracker Setting
TRACKER_CONFIG = "config/custom_bytetrack.yaml" #built_in Bytetrack config from Ultralytics

# --- Door zone settings ---
# Define the polygon points for the door zone (clockwise order).
# These were calibrated against ZONE_REFERENCE_RESOLUTION below.
DOOR_ZONE_POINTS: List[Tuple[int, int]] = [
    (638, 429),   # point 1 - fill in with your 4 real click results
    (251, 579),   # point 2
    (226, 105),   # point 3
    (528, 39),    # point 4
]

# The resolution you were looking at when you clicked the 4 points
# in find_coordinates_box.py.
ZONE_REFERENCE_RESOLUTION: Tuple[int, int] = (1920, 1200)


def get_zone_points() -> List[Tuple[int, int]]:
    """
    Return the door zone polygon scaled from its calibration resolution
    to the current PROCESS_(WIDTH|HEIGHT). If both resolutions match,
    the points are returned unchanged - so today this is a no-op, but it
    protects you if you later change the processing resolution.
    """
    ref_width, ref_height = ZONE_REFERENCE_RESOLUTION
    if (ref_width, ref_height) == (PROCESS_WIDTH, PROCESS_HEIGHT):
        return list(DOOR_ZONE_POINTS)

    scale_x = PROCESS_WIDTH / ref_width
    scale_y = PROCESS_HEIGHT / ref_height
    return [
        (int(round(x * scale_x)), int(round(y * scale_y)))
        for x, y in DOOR_ZONE_POINTS
    ]


# --- Decision-logic tuners ---
SIMILARITY_THRESHOLD = 0.85            # minimum similarity to call a match
TIME_WINDOW_SECONDS = 300              # within this, a match = DUPLICATE, else COUNT_AGAIN
FAST_MATCH_SECONDS = 5.0               # match faster than this is "risky"
FAST_MATCH_THRESHOLD = 0.93            # require this HIGHER score for fast matches

# --- Memory tuners ---
MEMORY_MAX_AGE_SECONDS = 7200          # how long to remember a person (2 hours)
MEMORY_MAX_VIEWS = 4                   # how many appearance views to keep per person

# --- Zone crossing ---
ZONE_COOLDOWN_SECONDS = 8.0            # debounce time per track_id at the zone edge

# --- UI / buffers ---
CROP_BUFFER_SIZE = 6                   # frames buffered per track_id for feature averaging
MAX_GALLERY_SIZE = 5                   # thumbnails in the gallery strip
THUMB_SIZE = 85                        # gallery thumbnail side length


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# --- Debug settings ---
DEBUG_MODE = _env_flag("DEBUG_MODE", False)  # True to see detailed decision logic logs