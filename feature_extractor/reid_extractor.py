# feature_extractor/reid_extractor.py

import os
import numpy as np
import cv2
from typing import Optional
from torchreid.reid.utils import FeatureExtractor as TorchreidExtractor

from config.settings import REID_MODEL_PATH


class FeatureExtractor:
    """
    This class turns a person's picture into a feature vector.
    Feature vector = many numbers that describe how the person looks.

    Why a class? Because loading the OSNet model is slow.
    We load it ONE time here, then reuse it many times.

    IMPORTANT FIX (see chat explanation):
    We use torchreid.utils.FeatureExtractor here, NOT
    torchreid.models.build_model(pretrained=True). build_model with
    pretrained=True loads generic ImageNet weights (trained to tell
    apart cats/dogs/cars - 1000 general object types). It was never
    trained to tell people apart. FeatureExtractor loads REAL Re-ID
    weights (the checkpoint path in config/settings.py), trained
    specifically on datasets of real people, to answer "are these two
    photos the same person?" - which is exactly the question our
    project needs answered.

    NOTE: if the checkpoint path points to a file that does NOT exist,
    TorchReID silently falls back to the generic ImageNet weights, so
    we fail fast with a clear error instead (see __init__).
    """

    def __init__(self, device: str = "cpu"):
        """
        Load the OSNet model one time, with correct Re-ID pretrained weights.

        We use osnet_x0_25 - a SMALL, FAST version of OSNet.
        Good for real-time use, even on CPU.
        """
        self.device = device

        if not os.path.exists(REID_MODEL_PATH):
            raise RuntimeError(
                f"Re-ID model file not found: {REID_MODEL_PATH}\n"
                "Expected an OSNet checkpoint in models/OsNetReID/ (e.g. "
                "osnet_x0_25_msmt17.pth). If missing, download it from the "
                "TorchReID model zoo and place it there."
            )

        self.torchreid_extractor = TorchreidExtractor(
            model_name="osnet_x0_25", # Small, fast OSNet model. Good for real-time use, even on CPU.
            model_path=REID_MODEL_PATH, # Real Re-ID weights, trained on real people datasets.
            device=device
        )

    def extract(self, cropped_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Turn one crop picture into a feature vector.

        Input: a crop picture (numpy array, BGR color, from OpenCV)
        Output: a feature vector (numpy array of numbers), or None if input is bad

        Note: resizing to the model's expected input size, andpixel
        normalization, are handled internally by TorchreidExtractor.
        We only need to fix the color order (BGR -> RGB) ourselves,
        since that's specific to how OpenCV read the image, not
        something the model's internal preprocessing can know about.
        """
        if cropped_image is None or cropped_image.size == 0:
            return None

        # OpenCV uses BGR color order. The model expects RGB.
        rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # TorchreidExtractor takes a LIST of images (built for batches).
        # We pass a list of one image, and read back row [0].
        features = self.torchreid_extractor([rgb_image])
        feature_vector = features[0].cpu().numpy()

        # Normalize - make vector length exactly 1.0.
        # This keeps the DIRECTION (appearance pattern) but removes
        # differences caused only by length, not identity.
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm

        return feature_vector

