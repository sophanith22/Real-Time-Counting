# utils/co_occurrence.py

from collections import defaultdict
from typing import Dict, List, Set


class CoOccurrenceTracker:
    """
    Remembers which track_ids were seen TOGETHER, in the same frame,
    at the same time.

    Why this matters: two different track_ids seen in the SAME frame
    can NEVER be the same real person - one person cannot be in two
    places at once. This is a simple, 100%-certain rule (no similarity
    score needed) that helps stop wrong matches in a crowd, especially
    when two people wear similar clothes.
    """

    def __init__(self):
        self.co_occurred: Dict[int, Set[int]] = defaultdict(set)

    def update(self, track_ids: List[int]) -> None:
        """
        Call this ONCE per frame, with the list of all track_ids
        currently visible in that frame.
        """
        n = len(track_ids)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = track_ids[i], track_ids[j]
                self.co_occurred[a].add(b)
                self.co_occurred[b].add(a)

    def get_co_occurred(self, track_id: int) -> Set[int]:
        """
        Return the set of track_ids that have ever appeared in the
        same frame as this track_id.
        """
        return self.co_occurred.get(track_id, set())