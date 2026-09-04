# detector/yolo_detector.py

from ultralytics import YOLO
import numpy as np
from typing import List, Tuple


class PersonDetector:
    """
    This class finds people in one video frame using YOLO.
    """

    def __init__(self, model_path: str, confidence_threshold: float,
                 person_class_id: int, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        results = self.model.predict(
            source=frame,
            device=self.device,
            verbose=False
        )

        detections = []
        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != self.person_class_id:
                continue
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, conf))

        return detections