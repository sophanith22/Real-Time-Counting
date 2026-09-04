# utils/line_crossing.py

from typing import Dict, Tuple


class LineCrossingDetector:
    """
    Watches tracked people and detects when they cross a virtual
    line SEGMENT (not an infinite line) - moving from OUTSIDE to
    INSIDE, only counting crossings that happen within the actual
    door/gate width defined by point_a and point_b.
    """

    def __init__(self, point_a: Tuple[int, int], point_b: Tuple[int, int],
                 inside_is_positive_side: bool = True):
        self.point_a = point_a
        self.point_b = point_b
        self.inside_is_positive_side = inside_is_positive_side
        self.last_side: Dict[int, str] = {}

    def _get_center_point(self, box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _cross_product_side(self, point: Tuple[int, int]) -> float:
        ax, ay = self.point_a
        bx, by = self.point_b
        px, py = point
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    def _get_side(self, point: Tuple[int, int]) -> str:
        cross_value = self._cross_product_side(point)
        if self.inside_is_positive_side:
            return "inside" if cross_value > 0 else "outside"
        else:
            return "inside" if cross_value < 0 else "outside"

    def _is_within_segment(self, point: Tuple[int, int]) -> bool:
        """
        NEW: Check if the person's position is actually BETWEEN
        point_a and point_b - not just on the correct SIDE of the
        infinite line. This stops false crossings from people
        walking left-right far away from the real door.

        We project the person's point onto the line A->B, and
        check if that projection falls between 0 (at A) and 1
        (at B). A small margin (-0.1 to 1.1) allows a little
        wiggle room at the edges, since people aren't perfect points.
        """
        ax, ay = self.point_a
        bx, by = self.point_b
        px, py = point

        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy

        if length_squared == 0:
            return False

        t = ((px - ax) * dx + (py - ay) * dy) / length_squared

        margin = 0.0
        return -margin <= t <= 1 + margin

    def check(self, track_id: int, box: Tuple[int, int, int, int]) -> bool:
        point = self._get_center_point(box)
        current_side = self._get_side(point)
        within_segment = self._is_within_segment(point)

        print(f"ID {track_id} at point {point}, side={current_side}, within_segment={within_segment}")

        if track_id not in self.last_side:
            self.last_side[track_id] = current_side
            return False

        previous_side = self.last_side[track_id]
        self.last_side[track_id] = current_side

        # NEW: only count as a real crossing if it happens WITHIN
        # the door's actual width, not somewhere far off to the side
        if previous_side == "outside" and current_side == "inside" and within_segment:
            return True

        return False