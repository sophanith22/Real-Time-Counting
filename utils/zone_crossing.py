# utils/zone_crossing.py

import time
from typing import Dict, Tuple, List

from config.log import get_logger
from config.settings import ZONE_COOLDOWN_SECONDS

log = get_logger("zone_crossing")


class ZoneCrossingDetector:
    """
    Watches tracked people and detects when they cross INTO a
    defined zone (polygon shape) - moving from OUTSIDE the zone
    to INSIDE it.

    Per our project scope: ENTRY ONLY.
    """

    def __init__(self, zone_points: List[Tuple[int, int]],
                 cooldown_seconds: float = ZONE_COOLDOWN_SECONDS):
        """
        zone_points: a list of (x, y) points defining the zone shape
        cooldown_seconds: after a track_id crosses, ignore any new
                           "crossed" trigger from the SAME track_id
                           for this many seconds. This stops box
                           jitter near the zone edge from counting
                           as many separate crossings.
        """
        self.zone_points = zone_points
        self.cooldown_seconds = cooldown_seconds
        self.last_state: Dict[int, str] = {}
        self.last_crossed_time: Dict[int, float] = {}

    def _get_feet_point(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, y2)

    def _point_in_polygon(self, point: Tuple[int, int]) -> bool:
        x, y = point
        n = len(self.zone_points)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.zone_points[i]
            xj, yj = self.zone_points[j]

            intersects = ((yi > y) != (yj > y)) and \
                         (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            if intersects:
                inside = not inside

            j = i

        return inside

    def get_state(self, track_id: int, box: Tuple[int, int, int, int]) -> Tuple[str, bool]:
        point = self._get_feet_point(box)
        currently_inside = self._point_in_polygon(point)
        current_state = "inside" if currently_inside else "outside"

        if track_id not in self.last_state:
            self.last_state[track_id] = current_state
            return current_state, False

        previous_state = self.last_state[track_id]
        self.last_state[track_id] = current_state

        raw_crossed = (previous_state == "outside" and current_state == "inside")

        if not raw_crossed:
            return current_state, False

        # --- Cooldown check: block re-trigger from the same track_id ---
        now = time.time()
        last_time = self.last_crossed_time.get(track_id, 0.0)
        time_since_last = now - last_time
        if time_since_last < self.cooldown_seconds:
            log.debug(
                f"track_id {track_id}: jitter blocked "
                f"({time_since_last:.2f}s since last real crossing, "
                f"cooldown={self.cooldown_seconds}s)"
            )
            return current_state, False

        log.debug(
            f"track_id {track_id}: REAL crossing accepted "
            f"({time_since_last:.2f}s since last, or first time)"
        )

        self.last_crossed_time[track_id] = now
        return current_state, True

    def check(self, track_id: int, box: Tuple[int, int, int, int]) -> bool:
        _, crossed = self.get_state(track_id, box)
        return crossed