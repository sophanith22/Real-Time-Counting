# memory/feature_memory.py

import time
import numpy as np
from typing import List, Dict


class FeatureMemory:
    """
    This class stores feature vectors for people already counted.

    Each record now stores a SMALL LIST of feature vectors, not just
    one. Why? Because the same real person can look very different
    from the front vs the back (viewpoint change). If we only keep
    one blended average, a new back-view crop may never match well
    against an old front-view average - causing a wrong "NEW" count
    for the same person.

    By keeping a few different views per person, a new crop only
    needs to match ONE of the saved views well, not the average of
    all of them.

    Record shape: { "track_id": ..., "features": [vec1, vec2, ...],
                     "time": when_last_seen }
    """

    def __init__(self, max_age_seconds: float = 7200, max_views: int = 4):
        """
        max_age_seconds: how long to keep a record before deleting it.
        max_views: how many different views to keep per person.
                   Older views get replaced first (oldest-out) once
                   this limit is reached, so memory does not grow
                   forever per person.
        """
        self.records: List[Dict] = []
        self.max_age_seconds = max_age_seconds
        self.max_views = max_views

    def add(self, track_id: int, feature_vector: np.ndarray) -> None:
        """
        Save a NEW person into memory, starting with one view.
        """
        record = {
            "track_id": track_id,
            "features": [feature_vector],
            "time": time.time()
        }
        self.records.append(record)

    def add_view(self, index: int, feature_vector: np.ndarray) -> None:
        """
        Add ANOTHER view to an already-known person (used when a
        DUPLICATE match happens - we learn what this person looks
        like from a new angle, which helps future matches).
        """
        if not (0 <= index < len(self.records)):
            return

        views = self.records[index]["features"]
        views.append(feature_vector)

        # Keep only the most recent `max_views` - drop the oldest
        if len(views) > self.max_views:
            self.records[index]["features"] = views[-self.max_views:]

    def update_time(self, index: int) -> None:
        """
        Update the timestamp of an existing record (used when we see
        the same person again - "refresh" how recently we saw them).
        """
        if 0 <= index < len(self.records):
            self.records[index]["time"] = time.time()

    def get_all(self) -> List[Dict]:
        """
        Return all current records. Also cleans out very old ones first.
        """
        self._remove_old_records()
        return self.records

    def _remove_old_records(self) -> None:
        """
        Delete records older than max_age_seconds.
        """
        now = time.time()
        self.records = [
            r for r in self.records
            if (now - r["time"]) < self.max_age_seconds
        ]

    def count(self) -> int:
        """
        Return how many records are currently stored.
        """
        return len(self.records)