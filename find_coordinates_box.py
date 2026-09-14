# find_coordinates.py

import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config.settings import VIDEO_SOURCE, PROCESS_WIDTH, PROCESS_HEIGHT

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Point {len(clicked_points)}: (x={x}, y={y})")
        else:
            print("Already have 4 points. Press 'r' to reset, or 'q' to quit.")


cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

cv2.namedWindow("Click to find coordinates")
cv2.setMouseCallback("Click to find coordinates", mouse_callback)

print("Click 4 points around the door area, in ORDER (going around the shape).")
print("Example order: top-left, top-right, bottom-right, bottom-left")
print("Press 'r' to reset points. Press 'q' to quit when done.\n")

while True:
    for _ in range(2):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        continue

    frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    for i, point in enumerate(clicked_points):
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{i+1}:{point}", (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw lines connecting the points, forming the zone shape
    if len(clicked_points) >= 2:
        for i in range(len(clicked_points) - 1):
            cv2.line(frame, clicked_points[i], clicked_points[i + 1], (0, 255, 255), 2)

    # If we have all 4, close the shape by connecting back to point 1
    if len(clicked_points) == 4:
        cv2.line(frame, clicked_points[3], clicked_points[0], (0, 255, 255), 2)

    cv2.imshow("Click to find coordinates", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        clicked_points.clear()
        print("Points reset.")

cap.release()
cv2.destroyAllWindows()

print("\nFinal points clicked:")
for i, point in enumerate(clicked_points):
    print(f"  Point {i+1}: {point}")