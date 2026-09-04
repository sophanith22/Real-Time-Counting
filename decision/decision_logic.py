# decision/decision_logic.py

from typing import Tuple, Optional
import numpy as np
import time

from memory.feature_memory import FeatureMemory
from decision.similarity import cosine_similarity
from config.settings import DEBUG_MODE


class DecisionEngine:
    """
    This class decides: is a person NEW, or already COUNTED?

    It uses two rules together:
    1. Similarity rule - does this look like someone we saved before?
    2. Time rule - was that match recent, or long ago?
    3. Fast-match rule - if match is very fast, need higher similarity
       (protects against two different people with similar clothes)
    """

    def __init__(self, memory: FeatureMemory,
                 similarity_threshold: float = 0.85,
                 time_window_seconds: float = 300,
                 fast_match_seconds: float = 5.0,
                 fast_match_threshold: float = 0.93):
        """
        memory: the FeatureMemory object that stores past people
        similarity_threshold: how similar counts as "same person"
        time_window_seconds: how soon counts as "too soon, don't count again"
                              Default: 300 seconds = 5 minutes
        fast_match_seconds: if match happens faster than this, it is risky -
                             two different people wearing similar clothes can
                             look alike when they cross close together
        fast_match_threshold: when match is "fast" (see above), require this
                               HIGHER similarity before saying DUPLICATEss
        """
        self.memory = memory
        self.similarity_threshold = similarity_threshold
        self.time_window_seconds = time_window_seconds
        self.fast_match_seconds = fast_match_seconds
        self.fast_match_threshold = fast_match_threshold

    def evaluate(self, track_id: int, feature_vector: np.ndarray) -> Tuple[str, Optional[int], Optional[int], Optional[float], Optional[float]]:
        records = self.memory.get_all()

        if DEBUG_MODE:
            print(f"  [DEBUG] Memory has {len(records)} records BEFORE this evaluation")

        # Step 1: memory is empty - automatically NEW
        if len(records) == 0:
            self.memory.add(track_id, feature_vector)
            return "NEW", None, None, None, None

        # Step 2: compare against every record, find the best match
        # Each record now has multiple views, so find the best view match
        best_score = -1.0
        best_index = -1

        for i, record in enumerate(records):
            record_features = record.get("features", [record.get("feature")])
            for stored_feature in record_features:
                score = cosine_similarity(feature_vector, stored_feature)
                if DEBUG_MODE:
                    print(f"  Compared to record {i} similarity = {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_index = i

        matched_record = records[best_index]
        time_since_seen = time.time() - matched_record["time"]

        # Step 3: decide which threshold to use - normal or strict
        if time_since_seen < self.fast_match_seconds:
            required_score = self.fast_match_threshold
            if DEBUG_MODE:
                print(f"  [DEBUG] FAST match ({time_since_seen:.2f}s) - using strict threshold {required_score}")
        else:
            required_score = self.similarity_threshold

        if DEBUG_MODE:
            print(f"  >>> FINAL CHECK: best_score={best_score:.4f}, required={required_score}, is_new={best_score < required_score}")

        # Step 4: not similar enough - NEW person
        if best_score < required_score:
            self.memory.add(track_id, feature_vector)
            return "NEW", None, None, best_score, None

        # Step 5: similar enough - check the time window
        if time_since_seen < self.time_window_seconds:
            # Too soon - this is the same visit, do not count again
            self.memory.add_view(best_index, feature_vector)
            self.memory.update_time(best_index)
            return "DUPLICATE", best_index, matched_record["track_id"], best_score, time_since_seen
        else:
            # Long time passed - treat as a new visit
            self.memory.add_view(best_index, feature_vector)
            self.memory.update_time(best_index)
            return "COUNT_AGAIN", best_index, matched_record["track_id"], best_score, time_since_seen