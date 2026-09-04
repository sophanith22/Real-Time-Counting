# tracker/person_tracker.py

from ultralytics import YOLO
import numpy as np
from typing import List, Tuple


class PersonTracker:
    """
    This class finds AND tracks people across video frames.

    Why one class doing both detection and tracking here,
    instead of reusing PersonDetector from Module 1?

    Because Ultralytics' .track() method does detection and
    tracking together, in one optimized step. Trying to split
    them apart would mean re-detecting twice, or fighting the
    library's design. So this class loads its own YOLO model,
    same as PersonDetector does, but calls a different method.
    """

    def __init__(self, model_path: str, confidence_threshold: float,
                 person_class_id: int, tracker_config: str,
                 device: str = "cpu"):
        """
        Load the YOLO model one time, when this object is created.

        tracker_config: name of the tracker settings file.
        "bytetrack.yaml" comes built-in with Ultralytics — we don't
        need to create this file ourselves.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.tracker_config = tracker_config
        self.device = device

    def track(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, int, float]]:
        """
        Find and track all people in one frame.

        Input: one video frame (numpy array)
        Output: a list of tuples (track_id, x1, y1, x2, y2, confidence)

        Why persist=True?
        This tells the tracker: "remember tracks from the last frame
        I saw." Without this, every frame would start fresh with no
        memory of past tracks — every person would get a new ID
        every single frame, which defeats the whole purpose.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False
        )

        tracks = []
        boxes = results[0].boxes
        
        # If no tracks exist yet (e.g. very first frame, or no one
        # on screen), boxes.id can be None. We must check this,
        # or the code crashes trying to read IDs that don't exist.
        assert boxes is not None
        if boxes.id is None:
            return tracks

        for box in boxes:  # type: ignore
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != self.person_class_id:
                continue
            if conf < self.confidence_threshold:
                continue

            track_id = int(box.id[0])  # type: ignore
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tracks.append((track_id, x1, y1, x2, y2, conf))

        return tracks