# utils/cropper.py

import numpy as np
from typing import Tuple, Optional

# Minimum size a crop must be, to be considered "good enough" to use.
# Too small = not enough detail for the ReID model to work well.
MIN_CROP_WIDTH = 40
MIN_CROP_HEIGHT = 80


def crop_person(frame: np.ndarray, box: Tuple[int, int, int, int],
                padding_ratio: float = 0.05) -> Optional[np.ndarray]:
    """
    Cuts out a person's picture from the full frame, using their box.

    box: (x1, y1, x2, y2)
    padding_ratio: extra space around the box as a fraction of its size.

    Returns the cropped image, or None if the box is invalid
    (this can happen at frame edges - see edge cases below), or
    if the crop is too small to be useful for feature extraction.
    """
    x1, y1, x2, y2 = box
    frame_height, frame_width = frame.shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1
    padding_x = int(box_width * padding_ratio)
    padding_y = int(box_height * padding_ratio)
    x1 -= padding_x
    y1 -= padding_y
    x2 += padding_x
    y2 += padding_y

    # Edge case: clamp the box so it never goes outside the frame.
    # Example: if a person is half-out of frame, x1 could be
    # negative, or x2 could be bigger than the frame width.
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    # Edge case: after clamping, check the box still makes sense.
    # If x2 <= x1 or y2 <= y1, the box has zero or negative size -
    # cropping this would fail or give an empty image.
    if x2 <= x1 or y2 <= y1:
        return None

    crop_width = x2 - x1
    crop_height = y2 - y1

    # NEW edge case: crop is too small to be useful.
    # A very small crop (person far away, or badly detected) gives
    # a weak, unreliable feature vector later in Module 4. Better
    # to skip it now than risk a wrong match in Module 5.
    if crop_width < MIN_CROP_WIDTH or crop_height < MIN_CROP_HEIGHT:
        return None

    cropped = frame[y1:y2, x1:x2]
    return cropped.copy()  # Return a copy to avoid referencing the origin.