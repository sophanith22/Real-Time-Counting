# config/settings.py

import torch

# Check one time: do we have a GPU(CUDA) available?
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"  # Force CPU for testing, change to "cuda" for GPU if available

# Path to the YOLO model file
YOLO_MODEL_PATH_NANNO = "models/yolov8n.pt" # Nano model, fastest but least accurate. Good for real-time use, but may miss detections.
YOLO_MODEL_PATH_SMALL = "models/yolov8s.pt" # Small model, good for real-time use, but less accurate than medium model.
YOLO_MODEL_PATH_MEDIUM = "models/yolov8m.pt" # Medium model, better accuracy but slower than small model. Not good for real-time use.
YOLO_MODEL_PATH_LARGE = "models/yolov8l.pt" # Large model, best accuracy but slowest. Not good for real-time use.

YOLO_MODEL_PATH = YOLO_MODEL_PATH_SMALL  # Use the small model for better speed, or YOLO_MODEL_PATH_NANNO for faster but less accurate detection. 

# Path to the Re-ID model file (OSNet) is specific re-ID of people, not gerneric Imagenet.
# REID_MODEL_PATH = "models/reid/osnet_x1_0_msmt17.pth" # is bigger model of OSNet, accuracy is better but slower. Not good for real-time use.
REID_MODEL_PATH = "models/reid/osnet_x0_25_msmt17.pth" # is smallest model of OSNet, accuracy is lower but faster. Good for real-time use.

# Minimum score to keep a detection (0.0 - 1.0)
YOLO_CONFIDENCE_THRESHOLD = 0.5  # 0.5 is a good balance between false positives and false negatives.

# Class ID for "person" in the YOLO model (usually 0 for COCO dataset)
YOLO_PERSON_CLASS_ID = 0

# VIDEO_SOURCE = "rtsp://admin:Hik123456@192.168.88.117:554/ch1/main/av_stream"  # for RTSP stream
VIDEO_SOURCE = "rtsp://admin:Hik123456@192.168.88.125:554/ch1/main/av_stream"  # for RTSP stream

# Tracker Setting
TRACKER_CONFIG = "config/custom_bytetrack.yaml" #built_in Bytetrack config from Ultralytics

# --- Door zone settings ---
# Define the polygon points for the door zone (clockwise order)
DOOR_ZONE_POINTS = [
    (606, 445),   # point 1 - fill in with your 4 real click results
    (288, 560),   # point 2
    (204, 125),   # point 3
    (545, 35),   # point 4
]
# --- Debug settings ---
# DEBUG_MODE = False  # False to hide detailed decision logic prints, True to show them
DEBUG_MODE = True   # True to see detailed decision logic prints