# find_coordinates.py
# Small tool: click on the video, and it prints the pixel position
# of each click. Use this to find the exact points for your
# diagonal door line.

import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config.settings import VIDEO_SOURCE

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    """
    This function runs every time something happens with the mouse.
    We only care about LEFT clicks - when that happens, save and
    print the (x, y) position.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)}: (x={x}, y={y})")


cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

cv2.namedWindow("Click to find coordinates")
cv2.setMouseCallback("Click to find coordinates", mouse_callback)

print("Click on the video where you want to mark a point.")
print("Press 'q' to quit when done.\n")

while True:
    for _ in range(2):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 810))  # Resize for better visibility, adjust as needed

    # Draw a small circle on every point already clicked,
    # so you can see where you've marked so far.
    for point in clicked_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{point}", (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw a line connecting the points, if we have 2 or more
    if len(clicked_points) >= 2:
        cv2.line(frame, clicked_points[0], clicked_points[1], (0, 255, 255), 2)

    cv2.imshow("Click to find coordinates", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal points clicked:")
for i, point in enumerate(clicked_points):
    print(f"  Point {i+1}: {point}")