# decision/similarity.py

import numpy as np


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.

    Returns a number between -1.0 and 1.0.
    1.0 = identical direction (very similar person)
    0.0 = no relation
    -1.0 = opposite direction (very different)

    Since our vectors are already normalized (length = 1.0, from
    Module 4), this formula simplifies to just a dot product.
    But we write the full formula here, so this function works
    correctly even if given a non-normalized vector by mistake.
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))