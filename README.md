# Real-Time People Counting System Using AI and Camera

An intelligent people counting system built for AI Farm Robotics, using computer vision and deep learning to accurately count unique individuals entering a monitored area — without double-counting the same person twice.

**Project Type:** AI Engineering Internship Project
**Author:** Yong Sophanith
**Company:** AI Farm Robotics
**Timeline:** July 2026 – September 2026

---

## Overview

Traditional people counters (e.g. simple motion sensors or basic tracking) often double-count people who briefly leave and re-enter a camera's view, or who get lost during tracking due to occlusion. This system solves that problem by combining detection, tracking, and appearance-based re-identification (Re-ID) into a single pipeline.

**Pipeline:**

```
Camera → YOLOv8 Detection → ByteTrack Tracking → Door Zone Detection
       → Person Cropping → Re-ID Feature Extraction → Memory & Decision Logic
       → Entry Count
```

Even if a tracked person's ID switches (a known limitation of motion-based trackers), the Re-ID + memory system recognizes them by appearance and prevents duplicate counting.

---

## Features

- Real-time person detection using YOLOv8 (pretrained, no custom training required)
- Multi-object tracking using ByteTrack, with tuned settings to reduce ID switching
- Configurable polygon door zone for entry detection
- Automatic crop-quality filtering (skips blurry or too-small detections)
- Multi-frame feature averaging for more stable person recognition
- Appearance-based Re-Identification using TorchReID (OSNet, trained on real person Re-ID data)
- Cosine similarity + time-window decision logic to prevent duplicate counting
- Live visual display: bounding boxes, per-person color coding, entry count, FPS, and a gallery strip of recently counted people
- CSV session logging for every entry event and end-of-session summary
- GPU (CUDA) acceleration support, with automatic CPU fallback

---

## Project Architecture

```
People_Counting_Project/
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # Model paths, thresholds, video source, and door zone
│   └── custom_bytetrack.yaml    # Tuned ByteTrack tracker settings
│
├── detector/
│   ├── __init__.py
│   ├── yolo_detector.py         # PersonDetector — YOLOv8 wrapper
│   └── test_yolo_model_compare.py  # Compares YOLOv8 model sizes (n/s/m)
│
├── tracker/
│   ├── __init__.py
│   └── person_tracker.py        # PersonTracker — YOLO + ByteTrack combined
│
├── utils/
│   ├── __init__.py
│   ├── cropper.py               # crop_person() — safe cropping with quality checks
│   ├── line_crossing.py         # Legacy straight-line detector
│   └── zone_crossing.py         # ZoneCrossingDetector — active entry logic
│
├── feature_extractor/
│   ├── __init__.py
│   └── reid_extractor.py        # FeatureExtractor — OSNet Re-ID feature vectors
│
├── memory/
│   ├── __init__.py
│   └── feature_memory.py        # FeatureMemory — stores known people with timestamps
│
├── decision/
│   ├── __init__.py
│   ├── similarity.py            # cosine_similarity()
│   ├── decision_logic.py        # DecisionEngine — NEW / DUPLICATE / COUNT_AGAIN logic
│   ├── test_decision.py         # Ground truth accuracy test
│   └── test_threshold_sweep.py  # Automated threshold sweep test
│
├── models/
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   ├── yolo11m.pt
│   └── OsNetReID/
│       └── osnet_x0_25_msmt17.pth
│
├── output/
│   ├── crops/                   # Saved images of counted people
│   └── logs/                    # CSV session logs
│
├── videos/                      # Test video files
├── main.py                      # Full system entry point
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.12 (3.14 is not yet compatible with required package wheels)

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

Then copy `.env.example` to `.env` and set your camera URL:

```bash
Copy-Item .env.example .env     # then edit .env
```

### GPU Support (recommended)

By default, `pip install torch` installs a CPU-only build. For CUDA/GPU acceleration on an NVIDIA GPU:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

The system automatically detects and uses CUDA if available, falling back to CPU otherwise — no code changes needed. See [CPU vs GPU Performance](#cpu-vs-gpu-performance) below for measured results.

---

## Configuration

All settings live in `config/settings.py`:

| Setting                     | Purpose                                                    |
| --------------------------- | ---------------------------------------------------------- |
| `VIDEO_SOURCE`              | Read from `.env` — RTSP stream, video file path, or `0` for webcam |
| `PROCESS_WIDTH/HEIGHT`      | Resolution every frame is resized to before processing     |
| `DOOR_ZONE_POINTS`          | Four or more `(x, y)` points that define the entry polygon |
| `ZONE_REFERENCE_RESOLUTION` | Resolution the polygon was drawn against (1920×1200)       |
| `YOLO_MODEL_PATH`           | Which YOLO model file to use (overridable via `.env`)      |
| `YOLO_CONFIDENCE_THRESHOLD` | Minimum person detection confidence (currently `0.5`)      |
| `MEMORY_MAX_AGE_SECONDS`    | How long a person stays in memory before being forgotten   |
| `SIMILARITY_THRESHOLD`      | Cosine-similarity cutoff for "same person" (currently `0.85`) |
| `DEVICE`                    | Automatically uses CUDA when available, otherwise CPU      |
| `DEBUG_MODE`                | Set via `.env`; toggles verbose decision-logic logs        |

**Camera credentials are loaded from `.env`, not from source files.** Copy `.env.example` to `.env`, fill in your RTSP URL, and the system picks it up automatically. This keeps passwords out of source control.

**Important:** `DOOR_ZONE_POINTS` must be set again for every new camera or video. The points must match the door area in the processed resolution defined by `PROCESS_WIDTH`/`PROCESS_HEIGHT` (1920×1200). If you change the processing resolution later, `get_zone_points()` automatically rescales the polygon from `ZONE_REFERENCE_RESOLUTION`. Use `find_coordinates_box.py` to click the 4 polygon corners on the live view. The detector counts movement from outside the polygon to inside it; it does not count exits.

---

## Usage

```bash
python main.py
```

- Press `q` to stop the session.
- A CSV log is saved automatically to `output/logs/session_<timestamp>.csv`, containing detected entry events and a final summary (total crossing events, unique people counted).
- Counted person crops are saved to `output/crops/`.

---

## Module Summary

| Module                      | Description                                             | Status     |
| --------------------------- | ------------------------------------------------------- | ---------- |
| 1. Detection                | YOLOv8 person detection, tested on webcam and video     | ✅ Complete |
| 2. Tracking                 | ByteTrack ID assignment, tuned to reduce ID switching   | ✅ Complete |
| 3. Door Zone & Cropping     | Polygon entry zone, feet-point checking, crop filtering | ✅ Complete |
| 4. Re-ID Feature Extraction | OSNet-based 512-dim appearance feature vectors          | ✅ Complete |
| 5. Memory & Decision Logic  | Cosine similarity + time-window duplicate prevention    | ✅ Complete |

---

## Testing & Validation

### Ground Truth Accuracy Test

Validated against a hand-labeled set of 12 photos representing 6 real, distinct individuals (2 photos each):

- **Result: 6/6 people correctly identified — 100% accuracy**
- 6 correctly marked `NEW`, 6 correctly marked `DUPLICATE`, with correct matching pairs

### Threshold Sweep (photo-pair test)

Tested similarity thresholds from 0.70 to 0.95 across 13 labeled same/different pairs. Thresholds 0.75–0.95 all achieved 100% accuracy, with a clear score separation (same-person: 0.97–0.99, different-person: 0.51–0.74).

### Threshold Sensitivity Analysis (full real-video test)

To validate the threshold choice under real operating conditions (not just isolated photo pairs), the same 18-crossing video session (6 real people, each crossing the door zone 3 times) was run end-to-end through `main.py` at seven different threshold values:

| Threshold | Unique People Counted | Result                                               |
| --------- | --------------------- | ---------------------------------------------------- |
| 0.75      | 4                     | ❌ Incorrect — distinct people wrongly merged         |
| 0.80      | 5                     | ❌ Incorrect — still merging some people              |
| 0.85      | 6                     | ✅ Correct                                            |
| 0.90      | 6                     | ✅ Correct                                            |
| 0.95      | 6                     | ✅ Correct                                            |
| 0.99      | 6                     | ✅ Correct                                            |
| 1.00      | 18                    | ❌ Incorrect — every crossing treated as a new person |

**This produces a clear U-shaped error pattern:**
- **Too low** (≤ 0.80): different people are incorrectly merged into one identity, undercounting.
- **Too high** (1.00): no two real photos are ever pixel-perfect identical, so the system fails to recognize even the exact same returning person, drastically overcounting.
- **0.85–0.99**: a wide, stable, correct range — confirming 0.85 is not a fragile boundary value but sits comfortably within a robust safety margin.

**Final threshold selected: 0.85**, chosen at the lower edge of the verified safe range to maximize duplicate-detection sensitivity while remaining fully within proven-correct territory.

### Re-ID Model Comparison (OSNet size)

Compared `osnet_x0_25` (lightweight) vs `osnet_x1_0` (larger) — both achieved identical accuracy on the test set. `osnet_x0_25` was selected for the final system due to significantly faster inference with no accuracy trade-off observed.

### YOLOv8 Model Size Comparison

Compared three YOLOv8 model sizes on GPU, measuring real-time FPS and average person detections per frame on the same test video:

| Model            | FPS (GPU) | Avg. Detections / Frame |
| ---------------- | --------- | ----------------------- |
| YOLOv8n (nano)   | ~58.7     | ~4.70                   |
| YOLOv8s (small)  | ~59.3     | ~4.86                   |
| YOLOv8m (medium) | ~34.4     | ~5.45                   |

**Findings:** Nano and small performed almost identically in both speed and detection count — the extra capacity of "small" showed no meaningful benefit over "nano" on this GPU. Medium showed a real detection improvement (~16% more people detected per frame on average) but at a significant cost of ~42% lower FPS.

**Decision:** The benchmark favored YOLOv8n (nano) for speed, but the current configuration uses YOLOv8s (small) through `YOLO_MODEL_PATH`. YOLOv8m is a possible upgrade if future testing on busier, higher-occlusion scenes shows the smaller models missing detections.

### CPU vs GPU Performance

| Device                           | Frames Per Second (FPS)            |
| -------------------------------- | ---------------------------------- |
| CPU                              | ~12 FPS                            |
| GPU (NVIDIA RTX 3050, CUDA 12.5) | ~40 FPS average (range: 36–45 FPS) |

GPU acceleration provided approximately **3.3x** faster performance with YOLOv8n. For a single-camera, low-traffic deployment, CPU performance is sufficient for real-time counting; GPU is recommended if scaling to multiple simultaneous camera feeds, or if adopting a larger model (e.g. YOLOv8m) for higher accuracy.

---

## Known Limitations

- **Tracking ID switches:** ByteTrack may assign a new ID to the same person after occlusion or close overlap with another person. This is expected behavior and is compensated for by the Re-ID + memory system, not corrected at the tracking layer itself.
- **Door-zone sensitivity:** The polygon points are scene-specific and must be reconfigured for each new camera angle or door position. The current system counts entries only, from outside to inside.
- **Small model limitations:** `osnet_x0_25`, while fast, occasionally produces lower-confidence similarity scores on low-resolution or heavily cropped images — future testing with higher-resolution crops or a larger model could improve robustness in edge cases.
- **Detection model trade-off:** YOLOv8n may miss some detections in busy or heavily occluded scenes compared to YOLOv8m; this has not yet been tested under high crowd-density conditions.
- **Not yet tested on:** live CCTV feeds, high-crowd-density scenes, and significant lighting variation (planned for later testing phases).

---

## Technology Stack

- Python 3.12
- Ultralytics YOLOv8 (nano / small / medium compared)
- ByteTrack (via Ultralytics)
- TorchReID (OSNet)
- OpenCV
- PyTorch (CUDA-enabled, cu121 build)
- NumPy

---

## Future Work

- Full integration testing under harder real-world conditions (similar clothing, pose variation, occlusion, crowding)
- Re-test YOLOv8n vs YOLOv8m specifically under high-occlusion / crowded scenes to confirm whether the medium model's extra detections translate into fewer missed counts
- Testing with real CCTV camera footage
- Exit counting (bidirectional flow) as an extension beyond current entry-only scope
- Web dashboard for live monitoring (outside current project scope)